"""Common data types and the base class for market data feeds."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime


def _now() -> datetime:
    return datetime.now(UTC)


@dataclass
class Quote:
    """A point-in-time price quote."""

    symbol: str
    price: float
    timestamp: datetime = field(default_factory=_now)
    bid: float = 0.0
    ask: float = 0.0
    volume: float = 0.0
    change_24h: float = 0.0  # Percent change over the last 24h


@dataclass
class OHLCV:
    """A single candlestick bar."""

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


class DataFeed(ABC):
    """Base class for all market data feeds."""

    name: str = "base"

    @abstractmethod
    async def get_price(self, symbol: str) -> Quote | None:
        """Return the latest quote for a symbol, or None if unavailable."""

    @abstractmethod
    async def get_ohlcv(self, symbol: str, days: int = 30) -> list[OHLCV]:
        """Return historical candles for a symbol (empty list on error)."""
