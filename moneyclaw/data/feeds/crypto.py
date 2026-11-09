"""Crypto market data via the free CoinGecko API (no key needed)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import httpx
import structlog

from moneyclaw.data.feeds.base import OHLCV, DataFeed, Quote

log = structlog.get_logger()

COINGECKO_API = "https://api.coingecko.com/api/v3"


class CryptoFeed(DataFeed):
    """CoinGecko-backed crypto feed. Symbols are CoinGecko ids (e.g. "bitcoin")."""

    name = "coingecko"

    def __init__(self, base_url: str = COINGECKO_API, timeout: float = 10.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    async def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        """GET a JSON endpoint. Returns parsed JSON, or None on any error."""
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(f"{self._base_url}{path}", params=params)
                resp.raise_for_status()
                return resp.json()
        except Exception:
            log.warning("crypto_feed.request_failed", path=path, exc_info=True)
            return None

    async def get_price(self, symbol: str) -> Quote | None:
        data = await self._get(
            "/simple/price",
            {
                "ids": symbol,
                "vs_currencies": "usd",
                "include_24hr_vol": "true",
                "include_24hr_change": "true",
            },
        )
        if not isinstance(data, dict) or symbol not in data:
            return None
        entry = data[symbol]
        price = entry.get("usd")
        if price is None:
            return None
        return Quote(
            symbol=symbol,
            price=float(price),
            volume=float(entry.get("usd_24h_vol") or 0.0),
            change_24h=float(entry.get("usd_24h_change") or 0.0),
        )

    async def get_ohlcv(self, symbol: str, days: int = 30) -> list[OHLCV]:
        data = await self._get(f"/coins/{symbol}/ohlc", {"vs_currency": "usd", "days": days})
        if not isinstance(data, list):
            return []
        bars: list[OHLCV] = []
        for row in data:
            if not isinstance(row, (list, tuple)) or len(row) < 5:
                continue
            ts, o, h, low, c = row[:5]
            bars.append(
                OHLCV(
                    timestamp=datetime.fromtimestamp(ts / 1000, tz=UTC),
                    open=float(o),
                    high=float(h),
                    low=float(low),
                    close=float(c),
                )
            )
        return bars

    async def get_top_movers(self, limit: int = 10) -> list[Quote]:
        """Top coins by market cap, sorted by absolute 24h change."""
        data = await self._get(
            "/coins/markets",
            {"vs_currency": "usd", "order": "market_cap_desc", "per_page": 100, "page": 1},
        )
        if not isinstance(data, list):
            return []
        quotes = [
            Quote(
                symbol=c.get("id", ""),
                price=float(c.get("current_price") or 0.0),
                volume=float(c.get("total_volume") or 0.0),
                change_24h=float(c.get("price_change_percentage_24h") or 0.0),
            )
            for c in data
            if isinstance(c, dict)
        ]
        quotes.sort(key=lambda q: abs(q.change_24h), reverse=True)
        return quotes[:limit]

    async def get_funding_rates(self) -> list[dict[str, Any]]:
        """Perpetual funding rates from CoinGecko's derivatives endpoint."""
        data = await self._get("/derivatives")
        if not isinstance(data, list):
            return []
        rates = []
        for d in data:
            if not isinstance(d, dict) or d.get("funding_rate") is None:
                continue
            rates.append(
                {
                    "symbol": d.get("symbol", ""),
                    "market": d.get("market", ""),
                    "funding_rate": d["funding_rate"],
                    "index_price": d.get("index", 0),
                    "open_interest": d.get("open_interest", 0),
                }
            )
        return rates

    async def search(self, query: str) -> list[dict[str, Any]]:
        """Search coins by name/symbol. Returns [{id, symbol, name}, ...]."""
        data = await self._get("/search", {"query": query})
        if not isinstance(data, dict):
            return []
        return [
            {"id": c.get("id"), "symbol": c.get("symbol"), "name": c.get("name")}
            for c in data.get("coins", [])
        ]
