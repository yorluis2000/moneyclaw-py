"""Gold swing-trading signals (黄金做T策略).

Watches the GLD gold ETF and flags buy-low / sell-high moments using
RSI and support/resistance from recent daily bars. With a stock executor it trades via Alpaca
(dollar-based, budget-capped; sells close the whole position). Without one it is
advisory and keeps a paper position.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import structlog

from moneyclaw.data.feeds.stocks import StockFeed
from moneyclaw.data.indicators import rsi
from moneyclaw.execution.stock_signals import execute_stock_signal
from moneyclaw.plugins.base import Opportunity, Result, Score, Strategy, load_strategy_config

if TYPE_CHECKING:
    from moneyclaw.execution.alpaca import AlpacaExecutor

log = structlog.get_logger()

DEFAULT_CONFIG: dict[str, Any] = {
    "symbol": "GLD",  # SPDR Gold Shares ETF (tradable on Alpaca)
    "min_amount": 1.0,
    "max_amount": 20.0,
    "position_ratio": 0.5,
    "max_daily_trades": 5,
    "rsi_period": 14,
    "rsi_oversold": 30,
    "rsi_overbought": 70,
    "stop_loss_pct": 0.02,
    "take_profit_pct": 0.015,
    "lookback_days": 60,
    "cooling_period": 300,  # seconds between signals
}


class GoldTradingStrategy(Strategy):
    """RSI + support/resistance swing signals on gold."""

    name = "gold_trading"
    description = "黄金日内做T策略，通过高抛低吸获取短线收益 (GLD)"
    risk_level = "medium"
    min_llm_layer = 2

    def __init__(
        self,
        feed: StockFeed | None = None,
        stock_executor: AlpacaExecutor | None = None,
        config: dict[str, Any] | None = None,
    ):
        file_cfg = load_strategy_config(GoldTradingStrategy).get("parameters", {}) or {}
        self.config = {**DEFAULT_CONFIG, **file_cfg, **(config or {})}
        self._feed = feed or StockFeed()
        self._executor = stock_executor
        self.paper_position = 0.0  # USD notionally held
        self.today_trades = 0
        self._trade_day = datetime.now(UTC).date()
        self.last_signal_time: datetime | None = None

    def _held(self) -> float:
        """USD currently held in the symbol (real/dry-run position, or paper)."""
        if self._executor:
            pos = self._executor.get_position(self.config["symbol"])
            return pos.market_value if pos else 0.0
        return self.paper_position

    def _roll_day(self) -> None:
        today = datetime.now(UTC).date()
        if today != self._trade_day:
            self._trade_day = today
            self.today_trades = 0

    async def scan(self) -> list[Opportunity]:
        self._roll_day()
        cfg = self.config
        if self.today_trades >= cfg["max_daily_trades"]:
            return []
        if self.last_signal_time:
            elapsed = (datetime.now(UTC) - self.last_signal_time).total_seconds()
            if elapsed < cfg["cooling_period"]:
                return []

        bars = await self._feed.get_ohlcv(cfg["symbol"], days=cfg["lookback_days"])
        closes = [b.close for b in bars]
        value = rsi(closes, cfg["rsi_period"])
        if value is None:
            return []

        price = closes[-1]
        if self._executor:
            self._executor.mark_price(cfg["symbol"], price)
        held = self._held()
        window = bars[-cfg["rsi_period"] :]
        support = min(b.low for b in window)
        resistance = max(b.high for b in window)

        if held <= 0 and (value < cfg["rsi_oversold"] or price <= support * 1.005):
            side = "buy"
            amount = max(cfg["min_amount"], cfg["max_amount"] * cfg["position_ratio"])
            target = price * (1 + cfg["take_profit_pct"])
            stop = price * (1 - cfg["stop_loss_pct"])
        elif held > 0 and (value > cfg["rsi_overbought"] or price >= resistance * 0.995):
            side = "sell"
            amount = held
            target = price * (1 - cfg["take_profit_pct"])
            stop = price * (1 + cfg["stop_loss_pct"])
        else:
            return []

        if side == "buy" and self._executor:
            amount = min(amount, self._executor.max_buy_amount())
        if side == "buy" and amount < 1.0:
            return []  # No budget left — don't emit unexecutable signals

        self.last_signal_time = datetime.now(UTC)
        opp = Opportunity(
            strategy_name=self.name,
            title=f"Gold {side.upper()} @ {price:.2f} (RSI {value:.0f})",
            money_involved=amount,
            data={
                "symbol": cfg["symbol"],
                "signal_type": side,
                "side": side,
                "price": price,
                "rsi": value,
                "support": support,
                "resistance": resistance,
                "target_price": target,
                "stop_loss": stop,
            },
        )
        opp.pre_score = (await self.evaluate(opp)).value
        return [opp]

    async def evaluate(self, opp: Opportunity) -> Score:
        cfg = self.config
        d = opp.data
        price, target, stop = d.get("price", 0), d.get("target_price", 0), d.get("stop_loss", 0)
        if not price or not target or not stop:
            return Score(value=0.0, reasoning="incomplete signal data")

        reward, risk = abs(target - price), abs(stop - price)
        rr = reward / risk if risk else 0.0
        value = d.get("rsi", 50.0)
        if d.get("signal_type") == "buy":
            strength = max(0.0, (cfg["rsi_oversold"] - value) / cfg["rsi_oversold"])
        else:
            strength = max(0.0, (value - cfg["rsi_overbought"]) / (100 - cfg["rsi_overbought"]))

        score = 0.4 + 0.4 * min(strength * 2, 1.0) + 0.2 * min(rr / 2, 1.0)
        return Score(
            value=round(min(score, 1.0), 2),
            threshold=0.5,
            reasoning=f"RSI {value:.1f}, reward/risk {rr:.2f}",
        )

    async def execute(self, opp: Opportunity) -> Result:
        """Trade via the stock executor if configured; otherwise advisory (paper position)."""
        side = opp.data.get("signal_type")
        if self._executor:
            result = await execute_stock_signal(self._executor, opp)
            if result.success:
                self.today_trades += 1
            return result
        if side == "buy":
            self.paper_position += opp.money_involved
        elif side == "sell":
            self.paper_position = max(0.0, self.paper_position - opp.money_involved)
        self.today_trades += 1
        log.info("gold_trading.signal", side=side, amount=opp.money_involved)
        return Result(
            success=True,
            profit_loss=0.0,
            details={
                "action": "advisory",
                "message": (
                    f"Consider {side} ${opp.money_involved:,.0f} of gold near "
                    f"{opp.data.get('price', 0):.2f} (target {opp.data.get('target_price', 0):.2f},"
                    f" stop {opp.data.get('stop_loss', 0):.2f})"
                ),
                "paper_position": self.paper_position,
                **opp.data,
            },
        )

    def estimate_roi(self) -> float:
        return 1.05
