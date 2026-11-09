"""AI & Longevity Growth — rule-based swing trading on AI/tech and longevity/biotech stocks.

Buy low: price dips (RSI oversold or at/below the lower Bollinger band) while the
longer trend is intact (near/above its 100-day SMA) and short-term momentum turns up
(today's price above yesterday's close).

Sell high: take profit (+8%), stop loss (-7%), or overbought (RSI high or at the upper
band) while in profit.

Orders go through the Alpaca executor (dollar-based, budget-capped). Without an
executor the strategy is advisory and tracks paper positions. Scores are computed by
rules (pre_score), so no LLM is needed.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from statistics import fmean
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
    "watchlist": [
        "NVDA",
        "PLTR",
        "AMD",
        "AVGO",
        "PATH",
        "RXRX",
        "ISRG",
        "CRSP",
        "NTLA",
        "TEM",
        "LLY",
    ],
    "trade_amount": 10.0,
    "max_open_positions": 5,
    "lookback_days": 180,
    "rsi_period": 14,
    "rsi_oversold": 35,
    "rsi_overbought": 70,
    "bollinger_period": 20,
    "bollinger_std": 2,
    "trend_period": 100,
    "trend_tolerance": 0.10,
    "momentum_period": 10,
    "take_profit_pct": 0.08,
    "stop_loss_pct": 0.07,
    "signal_cooldown_minutes": 60,
    "min_score": 0.55,
}


class AiLongevityGrowth(Strategy):
    name = "ai_longevity_growth"
    description = "Swing-trade AI/tech & longevity biotech stocks (RSI + Bollinger + momentum)"
    risk_level = "high"  # volatile growth and small-cap biotech names
    min_llm_layer = 0  # rule-based

    def __init__(
        self,
        feed: StockFeed | None = None,
        stock_executor: AlpacaExecutor | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        file_cfg = {
            k: v for k, v in load_strategy_config(AiLongevityGrowth).items() if k != "enabled"
        }
        self.config = {**DEFAULT_CONFIG, **file_cfg, **(config or {})}
        self.config["watchlist"] = [s.upper() for s in self.config["watchlist"]]
        self._feed = feed or StockFeed()
        self._executor = stock_executor
        self._paper: dict[str, dict[str, float]] = {}  # symbol -> {entry, notional}
        self._last_signal: dict[str, datetime] = {}

    # --------------------------------------------------------------- positions

    def _entry_price(self, symbol: str) -> float | None:
        if self._executor:
            pos = self._executor.get_position(symbol)
            return pos.avg_entry_price if pos and pos.qty > 0 else None
        p = self._paper.get(symbol)
        return p["entry"] if p else None

    def _open_count(self) -> int:
        watch = set(self.config["watchlist"])
        if self._executor:
            return sum(1 for s in watch if self._executor.get_position(s))
        return len(self._paper)

    def _buy_capacity(self) -> float:
        amount = float(self.config["trade_amount"])
        if self._executor:
            amount = min(amount, self._executor.max_buy_amount())
        return round(amount, 2)

    # --------------------------------------------------------------- signals

    def _analyze(self, symbol: str, closes: list[float]) -> dict[str, Any] | None:
        c = self.config
        need = (
            max(c["trend_period"], c["bollinger_period"], c["rsi_period"], c["momentum_period"]) + 1
        )
        if len(closes) < need:
            return None
        r = rsi(closes, c["rsi_period"])
        bands = bollinger(closes, c["bollinger_period"], c["bollinger_std"])
        if r is None or bands is None:
            return None
        lower, mid, upper = bands
        price, prev = closes[-1], closes[-2]
        return {
            "symbol": symbol,
            "price": price,
            "prev_close": prev,
            "rsi": r,
            "bb_lower": lower,
            "bb_mid": mid,
            "bb_upper": upper,
            "sma_trend": fmean(closes[-c["trend_period"] :]),
            "momentum": price / closes[-1 - c["momentum_period"]] - 1,
        }

    def _buy_score(self, a: dict[str, Any]) -> float:
        c = self.config
        depth = max(0.0, (c["rsi_oversold"] - a["rsi"]) / c["rsi_oversold"])  # 0..1
        band = 1.0 if a["price"] <= a["bb_lower"] else 0.0
        bounce = min(max((a["price"] / a["prev_close"] - 1) / 0.02, 0.0), 1.0)  # +2% → full
        trend = 1.0 if a["price"] >= a["sma_trend"] else 0.5
        return round(
            min(1.0, 0.35 + 0.25 * min(depth * 3, 1.0) + 0.15 * band + 0.1 * bounce + 0.15 * trend),
            2,
        )

    def _signal(self, a: dict[str, Any]) -> Opportunity | None:
        c = self.config
        sym, price = a["symbol"], a["price"]
        entry = self._entry_price(sym)

        if entry:
            gain = price / entry - 1
            reason = None
            if gain <= -c["stop_loss_pct"]:
                reason, score = f"stop loss {gain:+.1%}", 1.0
            elif gain >= c["take_profit_pct"]:
                reason, score = f"take profit {gain:+.1%}", 0.95
            elif gain > 0 and (a["rsi"] > c["rsi_overbought"] or price >= a["bb_upper"]):
                reason, score = f"overbought (RSI {a['rsi']:.0f}) {gain:+.1%}", 0.85
            if not reason:
                return None
            return Opportunity(
                strategy_name=self.name,
                title=f"SELL {sym} @ {price:.2f} — {reason}",
                money_involved=0.0,
                pre_score=score,
                data={**a, "side": "sell", "entry_price": entry, "reason": reason},
            )

        dip = a["rsi"] < c["rsi_oversold"] or price <= a["bb_lower"]
        bounce = price > a["prev_close"]
        trend_ok = price >= a["sma_trend"] * (1 - c["trend_tolerance"])
        if not (dip and bounce and trend_ok):
            return None
        if self._open_count() >= c["max_open_positions"]:
            return None
        amount = self._buy_capacity()
        if amount < 1.0:
            return None
        score = self._buy_score(a)
        if score < c["min_score"]:
            return None
        reason = f"dip (RSI {a['rsi']:.0f}) bouncing, momentum {a['momentum']:+.1%}"
        return Opportunity(
            strategy_name=self.name,
            title=f"BUY {sym} ${amount:.2f} @ {price:.2f} — {reason}",
            money_involved=amount,
            pre_score=score,
            data={**a, "side": "buy", "reason": reason},
        )

    async def scan(self) -> list[Opportunity]:
        c = self.config
        now = datetime.now(UTC)
        cooldown = timedelta(minutes=c["signal_cooldown_minutes"])
        symbols = [
            s
            for s in c["watchlist"]
            if now - self._last_signal.get(s, datetime.min.replace(tzinfo=UTC)) >= cooldown
        ]
        if not symbols:
            return []

        results = await asyncio.gather(
            *(self._feed.get_ohlcv(s, days=c["lookback_days"]) for s in symbols),
            return_exceptions=True,
        )
        opps: list[Opportunity] = []
        for sym, bars in zip(symbols, results, strict=True):
            if isinstance(bars, BaseException) or not bars:
                continue
            closes = [b.close for b in bars]
            if self._executor:
                self._executor.mark_price(sym, closes[-1])
            analysis = self._analyze(sym, closes)
            if not analysis:
                continue
            opp = self._signal(analysis)
            if opp:
                self._last_signal[sym] = now
                opps.append(opp)

        # Sells first (free up budget), then the strongest buys
        opps.sort(key=lambda o: (o.data["side"] != "sell", -(o.pre_score or 0)))
        return opps

    async def evaluate(self, opp: Opportunity) -> Score:
        return Score(
            value=opp.pre_score or 0.0, threshold=0.5, reasoning=opp.data.get("reason", "")
        )

    async def execute(self, opp: Opportunity) -> Result:
        if self._executor:
            return await execute_stock_signal(self._executor, opp)
        # Advisory mode: track a paper position
        sym, side, price = opp.data["symbol"], opp.data["side"], opp.data["price"]
        pl = 0.0
        if side == "buy":
            self._paper[sym] = {"entry": price, "notional": opp.money_involved}
        elif sym in self._paper:
            p = self._paper.pop(sym)
            pl = round(p["notional"] * (price / p["entry"] - 1), 2)
        return Result(
            success=True,
            profit_loss=0.0,
            details={"action": "advisory", "paper_pl": pl, "message": opp.title, **opp.data},
        )

    def estimate_roi(self) -> float:
        return 1.1
