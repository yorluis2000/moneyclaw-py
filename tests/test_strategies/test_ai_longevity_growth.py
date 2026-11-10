"""Tests for the AI & Longevity Growth swing strategy."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from moneyclaw.config.settings import AlpacaSettings, RiskSettings
from moneyclaw.data.feeds.base import OHLCV
from moneyclaw.data.feeds.stocks import StockFeed
from moneyclaw.execution.alpaca import AlpacaExecutor
from moneyclaw.plugins.base import load_strategy_config
from strategies.ai_longevity_growth import AiLongevityGrowth


def _bars(closes: list[float]) -> list[OHLCV]:
    base = datetime(2025, 1, 1, tzinfo=UTC)
    return [
        OHLCV(timestamp=base + timedelta(days=i), open=c, high=c, low=c, close=c)
        for i, c in enumerate(closes)
    ]


UPTREND = [100 + 0.5 * i for i in range(150)]  # 100 → 174.5
# Uptrend, then a sharp 7-day dip, then a bounce day: classic buy-the-dip setup
DIP_BOUNCE = UPTREND + [UPTREND[-1] - 4 * k for k in range(1, 8)] + [UPTREND[-1] - 27]
DOWNTREND = [200 - 1.0 * i for i in range(150)]  # broken trend, no bounce


def _flat_then(last: float) -> list[float]:
    return [100.0] * 149 + [last]


def _strategy(series: dict[str, list[float]], executor=None, **cfg) -> AiLongevityGrowth:
    feed = StockFeed()
    feed.get_ohlcv = AsyncMock(side_effect=lambda s, days: _bars(series.get(s, [])))
    return AiLongevityGrowth(feed=feed, stock_executor=executor, config=cfg)


def _executor(**risk) -> AlpacaExecutor:
    settings = RiskSettings(**{"max_trade_amount": 20, "max_portfolio_budget": 50, **risk})
    return AlpacaExecutor(AlpacaSettings(), settings, dry_run=True)


class TestConfig:
    def test_watchlist(self) -> None:
        s = AiLongevityGrowth()
        assert s.config["watchlist"] == [
            "NVDA", "PLTR", "AMD", "AVGO", "PATH", "RXRX", "ISRG", "CRSP", "NTLA", "TEM", "LLY",
        ]  # fmt: skip
        assert s.name == "ai_longevity_growth"
        assert load_strategy_config(AiLongevityGrowth)["enabled"] is True


class TestBuySignals:
    async def test_buys_the_dip_in_uptrend(self) -> None:
        s = _strategy({"NVDA": DIP_BOUNCE}, watchlist=["NVDA"])
        opps = await s.scan()
        assert len(opps) == 1
        opp = opps[0]
        assert opp.data["side"] == "buy" and opp.data["symbol"] == "NVDA"
        assert opp.data["rsi"] < 35
        assert opp.money_involved == 10.0
        assert opp.pre_score is not None and opp.pre_score > 0.5

    async def test_no_buy_in_downtrend(self) -> None:
        s = _strategy({"PLTR": DOWNTREND}, watchlist=["PLTR"])
        assert await s.scan() == []

    async def test_no_buy_without_bounce(self) -> None:
        s = _strategy({"AMD": DIP_BOUNCE[:-1]}, watchlist=["AMD"])  # still falling
        assert await s.scan() == []

    async def test_no_buy_without_enough_history(self) -> None:
        s = _strategy({"TEM": DIP_BOUNCE[-30:]}, watchlist=["TEM"])
        assert await s.scan() == []

    async def test_buy_size_capped_by_remaining_budget(self) -> None:
        ex = _executor(max_portfolio_budget=50)
        await ex.buy("LLY", 20, price=100)
        await ex.buy("ISRG", 20, price=100)
        s = _strategy({"NVDA": DIP_BOUNCE}, executor=ex, watchlist=["NVDA"], trade_amount=10)
        opps = await s.scan()
        assert opps[0].money_involved == 10.0
        await ex.buy("CRSP", 10, price=100)  # budget now full
        s2 = _strategy({"AMD": DIP_BOUNCE}, executor=ex, watchlist=["AMD"])
        assert await s2.scan() == []  # no unexecutable signals when budget is used up

    async def test_max_open_positions(self) -> None:
        ex = _executor()
        await ex.buy("NVDA", 5, price=100)
        s = _strategy(
            {"AMD": DIP_BOUNCE}, executor=ex, watchlist=["NVDA", "AMD"], max_open_positions=1
        )
        opps = await s.scan()
        assert all(o.data["side"] != "buy" for o in opps)

    async def test_cooldown_prevents_repeat_signals(self) -> None:
        s = _strategy({"NVDA": DIP_BOUNCE}, watchlist=["NVDA"])
        assert len(await s.scan()) == 1
        assert await s.scan() == []

    async def test_feed_errors_are_skipped(self) -> None:
        feed = StockFeed()
        feed.get_ohlcv = AsyncMock(side_effect=RuntimeError("yfinance down"))
        s = AiLongevityGrowth(feed=feed, config={"watchlist": ["NVDA"]})
        assert await s.scan() == []


class TestSellSignals:
    @pytest.mark.parametrize(
        ("last", "reason"),
        [(110.0, "take profit"), (92.0, "stop loss")],
    )
    async def test_exit_rules(self, last: float, reason: str) -> None:
        ex = _executor()
        await ex.buy("NVDA", 10, price=100.0)
        s = _strategy({"NVDA": _flat_then(last)}, executor=ex, watchlist=["NVDA"])
        opps = await s.scan()
        assert len(opps) == 1
        assert opps[0].data["side"] == "sell"
        assert reason in opps[0].data["reason"]

    async def test_holds_inside_band(self) -> None:
        ex = _executor()
        await ex.buy("NVDA", 10, price=100.0)
        choppy = [100.0 + (0.5 if i % 2 else -0.5) for i in range(149)] + [100.4]
        s = _strategy({"NVDA": choppy}, executor=ex, watchlist=["NVDA"])
        assert await s.scan() == []  # +0.4%, neutral RSI, inside the bands → hold

    async def test_overbought_exit_only_in_profit(self) -> None:
        ex = _executor()
        await ex.buy("NVDA", 10, price=120.0)  # entry above the current price
        s = _strategy({"NVDA": UPTREND}, executor=ex, watchlist=["NVDA"])
        # Strong uptrend (RSI ~100) but price 174.5 → that's a profit, so it sells
        assert (await s.scan())[0].data["side"] == "sell"
        ex2 = _executor()
        await ex2.buy("NVDA", 10, price=180.0)  # underwater: -3%, no stop yet
        s2 = _strategy({"NVDA": UPTREND}, executor=ex2, watchlist=["NVDA"])
        assert await s2.scan() == []

    async def test_sells_sorted_before_buys(self) -> None:
        ex = _executor()
        await ex.buy("LLY", 10, price=100.0)
        s = _strategy(
            {"LLY": _flat_then(115.0), "NVDA": DIP_BOUNCE}, executor=ex, watchlist=["NVDA", "LLY"]
        )
        sides = [o.data["side"] for o in await s.scan()]
        assert sides == ["sell", "buy"]


class TestExecute:
    async def test_executes_through_alpaca_executor(self) -> None:
        ex = _executor()
        s = _strategy({"NVDA": DIP_BOUNCE}, executor=ex, watchlist=["NVDA"])
        opp = (await s.scan())[0]
        result = await s.execute(opp)
        assert result.success and result.details["dry_run"] is True
        assert result.details["side"] == "buy"
        assert "Bought NVDA $10.00" in result.details["message"]
        assert ex.get_position("NVDA") is not None
        assert ex.deployed_capital == pytest.approx(10.0)

    async def test_sell_reports_realized_pl(self) -> None:
        ex = _executor()
        await ex.buy("NVDA", 10, price=100.0)
        s = _strategy({"NVDA": _flat_then(110.0)}, executor=ex, watchlist=["NVDA"])
        result = await s.execute((await s.scan())[0])
        assert result.profit_loss == pytest.approx(1.0)
        assert ex.get_position("NVDA") is None

    async def test_advisory_without_executor(self) -> None:
        s = _strategy({"NVDA": DIP_BOUNCE}, watchlist=["NVDA"])
        result = await s.execute((await s.scan())[0])
        assert result.details["action"] == "advisory"
        assert "NVDA" in s._paper

    async def test_evaluate_passes_pre_score(self) -> None:
        s = _strategy({"NVDA": DIP_BOUNCE}, watchlist=["NVDA"])
        opp = (await s.scan())[0]
        assert (await s.evaluate(opp)).value == opp.pre_score


class TestStocksOnlyDefaults:
    def test_crypto_strategies_disabled(self) -> None:
        from strategies.crypto_dca import CryptoDCA
        from strategies.crypto_funding import CryptoFunding
        from strategies.crypto_price_alert import CryptoPriceAlert
        from strategies.smart_rebalance import SmartRebalance

        for cls in (CryptoDCA, CryptoFunding, CryptoPriceAlert, SmartRebalance):
            assert load_strategy_config(cls).get("enabled") is False, cls.__name__

    def test_gold_uses_gld(self) -> None:
        from strategies.gold_trading_strategy import GoldTradingStrategy

        assert GoldTradingStrategy().config["symbol"] == "GLD"


class TestZeroLLM:
    def test_layer_zero(self) -> None:
        assert AiLongevityGrowth.min_llm_layer == 0

    async def test_full_tick_with_no_llm_available(self) -> None:
        """Scan → score → trade end to end while every LLM call fails."""
        from moneyclaw.agent.brain import AgentBrain
        from moneyclaw.agent.evaluator import Evaluator
        from moneyclaw.execution.risk import RiskManager
        from moneyclaw.plugins.registry import StrategyRegistry

        ex = _executor()
        s = _strategy({"NVDA": DIP_BOUNCE}, executor=ex, watchlist=["NVDA"])
        registry = StrategyRegistry()
        await registry.register(s)
        llm = MagicMock()
        llm.complete = AsyncMock(side_effect=RuntimeError("No suitable model available"))
        memory = AsyncMock()
        memory.get_approved = AsyncMock(return_value=[])
        risk = RiskManager(ex._risk)
        risk.set_exposure_provider(lambda: ex.deployed_capital)
        brain = AgentBrain(
            llm=llm,
            memory=memory,
            planner=MagicMock(),
            evaluator=Evaluator(llm),
            strategies=registry,
            risk=risk,
            scheduler=MagicMock(),
            stock_executor=ex,
        )
        await brain._tick()
        llm.complete.assert_not_called()
        assert ex.get_position("NVDA") is not None


class TestDividendRuleScored:
    async def test_dividend_opportunities_are_pre_scored(self) -> None:
        from strategies.stock_dividend import StockDividend

        feed = StockFeed()
        feed.get_info = AsyncMock(return_value={"name": "Verizon", "dividend_yield": 0.065})
        opps = await StockDividend(feed=feed, watchlist=["VZ"]).scan()
        assert opps and opps[0].pre_score is not None
