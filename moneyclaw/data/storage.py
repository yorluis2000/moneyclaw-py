"""DuckDB-backed storage for market quotes and OHLCV candles.

Timestamps are stored as naive UTC and returned as timezone-aware UTC datetimes.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import duckdb

from moneyclaw.data.feeds.base import OHLCV, Quote

_SCHEMA = """
CREATE TABLE IF NOT EXISTS quotes (
    symbol     VARCHAR NOT NULL,
    ts         TIMESTAMP NOT NULL,
    price      DOUBLE NOT NULL,
    bid        DOUBLE DEFAULT 0,
    ask        DOUBLE DEFAULT 0,
    volume     DOUBLE DEFAULT 0,
    change_24h DOUBLE DEFAULT 0,
    PRIMARY KEY (symbol, ts)
);
CREATE TABLE IF NOT EXISTS ohlcv (
    symbol VARCHAR NOT NULL,
    ts     TIMESTAMP NOT NULL,
    open   DOUBLE NOT NULL,
    high   DOUBLE NOT NULL,
    low    DOUBLE NOT NULL,
    close  DOUBLE NOT NULL,
    volume DOUBLE DEFAULT 0,
    PRIMARY KEY (symbol, ts)
);
"""


def _to_db(ts: datetime) -> datetime:
    """Normalize to naive UTC for storage."""
    if ts.tzinfo is not None:
        ts = ts.astimezone(UTC)
    return ts.replace(tzinfo=None)


def _from_db(ts: datetime) -> datetime:
    return ts.replace(tzinfo=UTC)


class MarketStorage:
    """Persist and query market data in a local DuckDB file."""

    def __init__(self, db_path: str = "data/market.duckdb") -> None:
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = duckdb.connect(db_path)
        self._conn.execute(_SCHEMA)

    def close(self) -> None:
        self._conn.close()

    # --- Quotes ---

    def store_quotes(self, quotes: list[Quote]) -> int:
        if not quotes:
            return 0
        rows = [
            (q.symbol, _to_db(q.timestamp), q.price, q.bid, q.ask, q.volume, q.change_24h)
            for q in quotes
        ]
        self._conn.executemany("INSERT OR REPLACE INTO quotes VALUES (?, ?, ?, ?, ?, ?, ?)", rows)
        return len(rows)

    def query_prices(
        self,
        symbol: str,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int | None = None,
    ) -> list[Quote]:
        sql, params = self._range_query("quotes", symbol, start, end, limit)
        rows = self._conn.execute(
            sql.format(cols="symbol, ts, price, bid, ask, volume, change_24h"), params
        ).fetchall()
        return [
            Quote(symbol=s, timestamp=_from_db(ts), price=p, bid=b, ask=a, volume=v, change_24h=c)
            for s, ts, p, b, a, v, c in rows
        ]

    # --- OHLCV ---

    def store_ohlcv(self, symbol: str, bars: list[OHLCV]) -> int:
        if not bars:
            return 0
        rows = [
            (symbol, _to_db(b.timestamp), b.open, b.high, b.low, b.close, b.volume) for b in bars
        ]
        self._conn.executemany("INSERT OR REPLACE INTO ohlcv VALUES (?, ?, ?, ?, ?, ?, ?)", rows)
        return len(rows)

    def query_ohlcv(
        self,
        symbol: str,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int | None = None,
    ) -> list[OHLCV]:
        sql, params = self._range_query("ohlcv", symbol, start, end, limit)
        rows = self._conn.execute(
            sql.format(cols="ts, open, high, low, close, volume"), params
        ).fetchall()
        return [
            OHLCV(timestamp=_from_db(ts), open=o, high=h, low=lo, close=c, volume=v)
            for ts, o, h, lo, c, v in rows
        ]

    @staticmethod
    def _range_query(
        table: str,
        symbol: str,
        start: datetime | None,
        end: datetime | None,
        limit: int | None,
    ) -> tuple[str, list]:
        sql = f"SELECT {{cols}} FROM {table} WHERE symbol = ?"
        params: list = [symbol]
        if start is not None:
            sql += " AND ts >= ?"
            params.append(_to_db(start))
        if end is not None:
            sql += " AND ts <= ?"
            params.append(_to_db(end))
        sql += " ORDER BY ts"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))
        return sql, params
