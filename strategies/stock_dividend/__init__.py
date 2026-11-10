"""Stock Dividend Tracker — Layer 1 strategy.

Track high-dividend stocks, alert before ex-dividend dates.
Advisory only: sends buy suggestions, does not auto-trade stocks.
"""

from __future__ import annotations

import structlog

from moneyclaw.data.feeds.stocks import StockFeed
from moneyclaw.plugins.base import Opportunity, Result, Score, Strategy, load_strategy_config

log = structlog.get_logger()

# Default watchlist of high-dividend tickers
DEFAULT_WATCHLIST = ["VZ", "T", "MO", "XOM", "CVX", "PFE", "IBM", "KO", "PEP", "JNJ"]
MIN_DIVIDEND_YIELD = 0.04  # 4%


class StockDividend(Strategy):
    """Track high-dividend stocks and alert on buying opportunities.

    Layer 1: uses local LLM to evaluate fundamentals before recommending.
    Advisory only — sends suggestions, does NOT auto-trade stocks.
    """

    name = "stock_dividend"
    description = "Track high-dividend stocks, alert before ex-dividend dates"
    risk_level = "low"
    min_llm_layer = 1

    def __init__(
        self,
        feed: StockFeed | None = None,
        watchlist: list[str] | None = None,
        min_yield: float | None = None,
    ) -> None:
        cfg = load_strategy_config(StockDividend)
        self._feed = feed or StockFeed()
        self._watchlist = watchlist or cfg.get("watchlist", DEFAULT_WATCHLIST)
        self._min_yield = (
            min_yield if min_yield is not None else cfg.get("min_yield", MIN_DIVIDEND_YIELD)
        )
        self._alerted: set[str] = set()  # Track already-alerted tickers

    async def scan(self) -> list[Opportunity]:
        """Scan watchlist for stocks with high dividend yield."""
        opportunities = []

        for ticker in self._watchlist:
            if ticker in self._alerted:
                continue

            info = await self._feed.get_info(ticker)
            if not info:
                continue

            div_yield = info.get("dividend_yield") or 0
            if div_yield < self._min_yield:
                continue

            ex_date = info.get("ex_dividend_date")
            opp = Opportunity(
                strategy_name=self.name,
                title=f"Dividend: {ticker} yields {div_yield:.1%}",
                money_involved=0,  # Advisory only
                data={
                    "ticker": ticker,
                    "name": info.get("name", ""),
                    "dividend_yield": div_yield,
                    "pe_ratio": info.get("pe_ratio"),
                    "market_cap": info.get("market_cap"),
                    "ex_dividend_date": ex_date,
                    "sector": info.get("sector", ""),
                },
            )
            # Layer 0: rule-based score, so the agent never needs an LLM for this
            opp.pre_score = (await self.evaluate(opp)).value
            opportunities.append(opp)

        return opportunities

    async def evaluate(self, opp: Opportunity) -> Score:
        """Evaluate dividend opportunity based on yield and fundamentals."""
        div_yield = opp.data.get("dividend_yield", 0)
        pe = opp.data.get("pe_ratio")

        # Simple scoring: higher yield = higher score, penalize very high PE
        score = min(div_yield / 0.10, 1.0)  # 10%+ yield = max score
        if pe and pe > 30:
            score *= 0.7  # Penalize overvalued stocks

        return Score(
            value=score,
            threshold=0.4,
            reasoning=f"{opp.data.get('ticker')}: {div_yield:.1%} yield, PE={pe}",
        )

    async def execute(self, opp: Opportunity) -> Result:
        """'Execute' = mark as alerted. This is advisory, not auto-trading."""
        ticker = opp.data.get("ticker", "")
        self._alerted.add(ticker)
        return Result(
            success=True,
            profit_loss=0,
            details={
                "action": "advisory",
                "message": (
                    f"Consider buying {ticker}"
                    f" for {opp.data.get('dividend_yield', 0):.1%} dividend yield"
                ),
                **opp.data,
            },
        )

    def estimate_roi(self) -> float:
        """Average high-dividend stock: ~4-6% yield + modest appreciation."""
        return 1.05
