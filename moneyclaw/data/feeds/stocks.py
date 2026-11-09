"""Stock market data via yfinance (blocking calls run in a worker thread)."""

from __future__ import annotations

import asyncio
from datetime import UTC
from typing import Any

import structlog
import yfinance as yf

from moneyclaw.data.feeds.base import OHLCV, DataFeed, Quote

log = structlog.get_logger()


def _num(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


class StockFeed(DataFeed):
    """yfinance-backed stock feed. Symbols are tickers (e.g. "AAPL")."""

    name = "yfinance"

    async def get_price(self, symbol: str) -> Quote | None:
        def _fetch() -> Quote | None:
            fi = yf.Ticker(symbol).fast_info
            price = _num(getattr(fi, "last_price", None))
            if price is None:
                return None
            prev = _num(getattr(fi, "previous_close", None))
            change = (price - prev) / prev * 100 if prev else 0.0
            return Quote(
                symbol=symbol,
                price=price,
                volume=_num(getattr(fi, "last_volume", None)) or 0.0,
                change_24h=change,
            )

        try:
            return await asyncio.to_thread(_fetch)
        except Exception:
            log.warning("stock_feed.price_failed", symbol=symbol, exc_info=True)
            return None

    async def get_info(self, symbol: str) -> dict[str, Any] | None:
        """Fundamentals, mapped to snake_case keys."""

        def _fetch() -> dict[str, Any] | None:
            info = yf.Ticker(symbol).info
            if not info:
                return None
            return {
                "symbol": symbol,
                "name": info.get("shortName") or info.get("longName", ""),
                "sector": info.get("sector", ""),
                "pe_ratio": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "market_cap": info.get("marketCap"),
                "dividend_yield": info.get("dividendYield"),
                "ex_dividend_date": info.get("exDividendDate"),
                "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
                "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
            }

        try:
            return await asyncio.to_thread(_fetch)
        except Exception:
            log.warning("stock_feed.info_failed", symbol=symbol, exc_info=True)
            return None

    async def get_dividends(self, symbol: str) -> list[dict[str, Any]]:
        """Dividend history as [{date, amount}, ...], oldest first."""

        def _fetch() -> list[dict[str, Any]]:
            divs = yf.Ticker(symbol).dividends
            if divs is None or len(divs) == 0:
                return []
            return [
                {"date": ts.date().isoformat(), "amount": float(amount)}
                for ts, amount in divs.items()
            ]

        try:
            return await asyncio.to_thread(_fetch)
        except Exception:
            log.warning("stock_feed.dividends_failed", symbol=symbol, exc_info=True)
            return []

    async def get_ohlcv(self, symbol: str, days: int = 30) -> list[OHLCV]:
        def _fetch() -> list[OHLCV]:
            hist = yf.Ticker(symbol).history(period=f"{days}d")
            bars = []
            for ts, row in hist.iterrows():
                dt = ts.to_pydatetime()
                dt = dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)
                bars.append(
                    OHLCV(
                        timestamp=dt,
                        open=float(row["Open"]),
                        high=float(row["High"]),
                        low=float(row["Low"]),
                        close=float(row["Close"]),
                        volume=float(row.get("Volume", 0) or 0),
                    )
                )
            return bars

        try:
            return await asyncio.to_thread(_fetch)
        except Exception:
            log.warning("stock_feed.ohlcv_failed", symbol=symbol, exc_info=True)
            return []
