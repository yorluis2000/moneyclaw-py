"""Tests for the gold and AAPL swing-signal strategies and indicators."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest

from moneyclaw.data.feeds.base import OHLCV
from moneyclaw.data.feeds.stocks import StockFeed
from moneyclaw.data.indicators import bollinger, rsi
from strategies.appl_trading_strategy import ApplTradingStrategy
from strategies.gold_trading_strategy import GoldTradingStrategy


def _bars(closes: list[float]) -> list[OHLCV]:
    base = datetime(2025, 1, 1, tzinfo=UTC)
    return [
        OHLCV(timestamp=base + timedelta(days=i), open=c, high=c * 1.01, low=c * 0.99, close=c)
        for i, c in enumerate(closes)
    ]


FALLING = [100.0 - i for i in range(40)]  # steady decline → oversold
RISING = [100.0 + i for i in range(40)]  # steady rise → overbought
FLAT = [100.0 + (1 if i % 2 else -1) * 0.5 for i in range(40)]


class TestIndicators:
    def test_rsi_extremes(self) -> None:
        assert rsi(FALLING, 14) < 5
        assert rsi(RISING, 14) > 95
        assert 40 < rsi(FLAT, 14) < 60

    def test_rsi_insufficient_data(self) -> None:
        assert rsi([1.0, 2.0], 14) is None

    def test_bollinger(self) -> None:
        lower, mid, upper = bollinger([10.0] * 20, 20, 2)
        assert lower == mid == upper == 10.0
        assert bollinger([1.0], 20) is None


@pytest.mark.parametrize("cls", [GoldTradingStrategy, ApplTradingStrategy])
class TestSwingStrategies:
    async def test_buy_signal_when_oversold(self, cls) -> None:
        feed = StockFeed()
        s = cls(feed=feed)
        with patch.object(feed, "get_ohlcv", new_callable=AsyncMock, return_value=_bars(FALLING)):
            opps = await s.scan()
        assert len(opps) == 1
        assert opps[0].data["signal_type"] == "buy"
        assert opps[0].money_involved > 0
        score = await s.evaluate(opps[0])
        assert score.value >= score.threshold

    async def test_no_sell_without_position(self, cls) -> None:
        feed = StockFeed()
        s = cls(feed=feed)
        with patch.object(feed, "get_ohlcv", new_callable=AsyncMock, return_value=_bars(RISING)):
            assert await s.scan() == []

    async def test_execute_then_sell(self, cls) -> None:
        feed = StockFeed()
        s = cls(feed=feed, config={"cooling_period": 0})
        with patch.object(feed, "get_ohlcv", new_callable=AsyncMock, return_value=_bars(FALLING)):
            buy = (await s.scan())[0]
        result = await s.execute(buy)
        assert result.success and result.details["action"] == "advisory"
        assert s.paper_position > 0

        with patch.object(feed, "get_ohlcv", new_callable=AsyncMock, return_value=_bars(RISING)):
            opps = await s.scan()
        assert opps[0].data["signal_type"] == "sell"

    async def test_empty_data(self, cls) -> None:
        feed = StockFeed()
        s = cls(feed=feed)
        with patch.object(feed, "get_ohlcv", new_callable=AsyncMock, return_value=[]):
            assert await s.scan() == []

    async def test_daily_limit(self, cls) -> None:
        s = cls(feed=StockFeed(), config={"max_daily_trades": 0})
        assert await s.scan() == []

    def test_loads_config_and_meta(self, cls) -> None:
        s = cls()
        assert s.config["rsi_period"] == 14
        assert s.risk_level == "medium"
        assert s.estimate_roi() == 1.05
