"""Error metrics used across all forecasters (single source of truth)."""

from __future__ import annotations

import math

from forecasting.models import BacktestWindow


def smape(actual: tuple[float, ...], forecast: tuple[float, ...]) -> float:
    """Symmetric MAPE in [0, 2]; robust to zero actuals (intermittent demand)."""
    if len(actual) != len(forecast):
        msg = "actual/forecast length mismatch"
        raise ValueError(msg)
    if not actual:
        msg = "no observations"
        raise ValueError(msg)
    total = 0.0
    for a, f in zip(actual, forecast, strict=True):
        denom = (abs(a) + abs(f)) / 2.0
        if denom == 0:
            total += 0.0  # both zero: perfect match on a zero period
        else:
            total += abs(f - a) / denom
    return total / len(actual)


def rmse(actual: tuple[float, ...], forecast: tuple[float, ...]) -> float:
    if len(actual) != len(forecast):
        msg = "actual/forecast length mismatch"
        raise ValueError(msg)
    if not actual:
        msg = "no observations"
        raise ValueError(msg)
    return math.sqrt(
        sum((f - a) ** 2 for a, f in zip(actual, forecast, strict=True)) / len(actual)
    )


def bias(actual: tuple[float, ...], forecast: tuple[float, ...]) -> float:
    """Mean signed error; positive = systematic over-forecasting."""
    if len(actual) != len(forecast):
        msg = "actual/forecast length mismatch"
        raise ValueError(msg)
    if not actual:
        msg = "no observations"
        raise ValueError(msg)
    return sum(f - a for a, f in zip(actual, forecast, strict=True)) / len(actual)


def score_window(window: BacktestWindow) -> tuple[float, float, float]:
    """(sMAPE, RMSE, bias) for one window."""
    return (
        smape(window.actuals, window.forecasts),
        rmse(window.actuals, window.forecasts),
        bias(window.actuals, window.forecasts),
    )
