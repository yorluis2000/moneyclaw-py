"""Pure-Python technical indicators used by swing-trading strategies."""

from __future__ import annotations

from statistics import fmean, pstdev


def rsi(closes: list[float], period: int = 14) -> float | None:
    """Wilder's RSI of the last value. None if there isn't enough data."""
    if period <= 0 or len(closes) <= period:
        return None
    deltas = [b - a for a, b in zip(closes, closes[1:], strict=False)]
    avg_gain = fmean(max(d, 0.0) for d in deltas[:period])
    avg_loss = fmean(max(-d, 0.0) for d in deltas[:period])
    for d in deltas[period:]:
        avg_gain = (avg_gain * (period - 1) + max(d, 0.0)) / period
        avg_loss = (avg_loss * (period - 1) + max(-d, 0.0)) / period
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def bollinger(
    closes: list[float], period: int = 20, num_std: float = 2.0
) -> tuple[float, float, float] | None:
    """(lower, middle, upper) Bollinger bands of the last `period` values."""
    if period <= 0 or len(closes) < period:
        return None
    window = closes[-period:]
    mid = fmean(window)
    sd = pstdev(window)
    return mid - num_std * sd, mid, mid + num_std * sd
