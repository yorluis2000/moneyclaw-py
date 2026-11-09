"""AAPL swing-trading signals (APPL做T交易策略).

Watches Apple (AAPL) and flags buy-low / sell-high moments using RSI and Bollinger
bands on recent daily bars. With a stock executor it trades via Alpaca
(dollar-based, budget-capped; sells close the whole position). Without one it is
advisory and keeps a paper position.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import structlog

from moneyclaw.data.feeds.stocks import StockFeed
from moneyclaw.data.indicators import bollinger, rsi
from moneyclaw.execution.stock_signals import execute_stock_signal
from moneyclaw.plugins.base import Opportunity, Result, Score, Strategy, load_strategy_config

if TYPE_CHECKING:
    from moneyclaw.execution.alpaca import AlpacaExecutor

log = structlog.get_logger()

DEFAULT_CONFIG: dict[str, Any] = {
    "symbol": "AAPL",
    "base_amount": 10.0,
    "max_position": 20.0,
    "min_profit_percent": 0.005,
    "max_loss_percent": 0.02,
    "rsi_period": 14,
    "rsi_oversold": 30,
    "rsi_overbought": 70,
    "bollinger_period": 20,
    "bollinger_std": 2,
    "max_daily_trades": 5,
    "cooling_period": 300,  # seconds between signals
    "lookback_days": 90,
}


class ApplTradingStrategy(Strategy):
    """RSI + Bollinger band swing signals on AAPL."""

    name = "appl_trading"
    description = "APPL做T策略 - 监控苹果公司股票进行波段交易"
    risk_level = "medium"
    min_llm_layer = 3

    def __init__(
        self,
        feed: StockFeed | None = None,
        stock_executor: AlpacaExecutor | None = None,
        config: dict[str, Any] | None = None,
    ):
        file_cfg = load_strategy_config(ApplTradingStrategy).get("parameters", {}) or {}
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
        bands = bollinger(closes, cfg["bollinger_period"], cfg["bollinger_std"])
        if value is None or bands is None:
            return []

        lower, mid, upper = bands
        price = closes[-1]
        if self._executor:
            self._executor.mark_price(cfg["symbol"], price)
        held = self._held()

        if held < cfg["max_position"] and (value < cfg["rsi_oversold"] or price <= lower):
            side = "buy"
            amount = min(cfg["base_amount"], cfg["max_position"] - held)
            target = max(mid, price * (1 + cfg["min_profit_percent"]))
            stop = price * (1 - cfg["max_loss_percent"])
        elif held > 0 and (value > cfg["rsi_overbought"] or price >= upper):
            side = "sell"
            amount = held
            target = min(mid, price * (1 - cfg["min_profit_percent"]))
            stop = price * (1 + cfg["max_loss_percent"])
        else:
            return []

        if side == "buy" and self._executor:
            amount = min(amount, self._executor.max_buy_amount())
        if side == "buy" and amount < 1.0:
            return []  # No budget left — don't emit unexecutable signals

        self.last_signal_time = datetime.now(UTC)
        opp = Opportunity(
            strategy_name=self.name,
            title=f"{cfg['symbol']} {side.upper()} @ {price:.2f} (RSI {value:.0f})",
            money_involved=amount,
            data={
                "symbol": cfg["symbol"],
                "signal_type": side,
                "side": side,
                "price": price,
                "rsi": value,
                "bb_lower": lower,
                "bb_mid": mid,
                "bb_upper": upper,
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

        expected = abs(target - price) / price
        max_loss = abs(stop - price) / price
        rr = expected / max_loss if max_loss else 0.0
        value = d.get("rsi", 50.0)
        if d.get("signal_type") == "buy":
            strength = max(0.0, (cfg["rsi_oversold"] - value) / cfg["rsi_oversold"])
            band_hit = price <= d.get("bb_lower", 0)
        else:
            strength = max(0.0, (value - cfg["rsi_overbought"]) / (100 - cfg["rsi_overbought"]))
            band_hit = price >= d.get("bb_upper", float("inf"))

        score = 0.3 + 0.3 * min(strength * 2, 1.0) + 0.2 * band_hit + 0.2 * min(rr / 2, 1.0)
        return Score(
            value=round(min(score, 1.0), 2),
            threshold=0.5,
            reasoning=(
                f"RSI {value:.1f}, expected {expected:.2%}, reward/risk {rr:.2f},"
                f" band {'hit' if band_hit else 'not hit'}"
            ),
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
        log.info("appl_trading.signal", side=side, amount=opp.money_involved)
        return Result(
            success=True,
            profit_loss=0.0,
            details={
                "action": "advisory",
                "message": (
                    f"Consider {side} ${opp.money_involved:,.0f} of {opp.data.get('symbol')} near "
                    f"{opp.data.get('price', 0):.2f} (target {opp.data.get('target_price', 0):.2f},"
                    f" stop {opp.data.get('stop_loss', 0):.2f})"
                ),
                "paper_position": self.paper_position,
                **opp.data,
            },
        )

    def estimate_roi(self) -> float:
        return 1.05
