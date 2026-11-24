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
    return math.sqrt(sum((f - a) ** 2 for a, f in zip(actual, forecast, strict=True)) / len(actual))


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


def _validate_scaled(
    actual: tuple[float, ...], forecast: tuple[float, ...], insample: tuple[float, ...]
) -> None:
    if len(actual) != len(forecast):
        msg = "actual/forecast length mismatch"
        raise ValueError(msg)
    if not actual:
        msg = "no observations"
        raise ValueError(msg)
    if len(insample) < 2:
        msg = "insample needs >= 2 observations to scale errors"
        raise ValueError(msg)


def _naive_abs(insample: tuple[float, ...]) -> float:
    return sum(abs(insample[i] - insample[i - 1]) for i in range(1, len(insample)))


def _naive_sq(insample: tuple[float, ...]) -> float:
    return sum((insample[i] - insample[i - 1]) ** 2 for i in range(1, len(insample)))


def mase(
    actual: tuple[float, ...], forecast: tuple[float, ...], insample: tuple[float, ...]
) -> float:
    """Mean absolute scaled error: MAE / in-sample naive MAE (m=1).

    Scale-free across SKUs, so a lumpy SKU's error reads next to a smooth
    one. `inf` when the in-sample naive error is zero (perfectly flat
    history) — an undefined scale, never a crash.
    """
    _validate_scaled(actual, forecast, insample)
    denom = _naive_abs(insample)
    if denom == 0:
        return math.inf
    mean_abs_err = sum(abs(f - a) for a, f in zip(actual, forecast, strict=True)) / len(actual)
    mean_naive = denom / (len(insample) - 1)
    return mean_abs_err / mean_naive


def rmsse(
    actual: tuple[float, ...], forecast: tuple[float, ...], insample: tuple[float, ...]
) -> float:
    """Root mean squared scaled error: RMSE / in-sample naive RMSE (m=1)."""
    _validate_scaled(actual, forecast, insample)
    denom = _naive_sq(insample)
    if denom == 0:
        return math.inf
    mean_sq_err = sum((f - a) ** 2 for a, f in zip(actual, forecast, strict=True)) / len(actual)
    mean_naive_sq = denom / (len(insample) - 1)
    return math.sqrt(mean_sq_err / mean_naive_sq)


def _quantile(sorted_values: list[float], q: float) -> float:
    """Linear-interpolation quantile (numpy's default)."""
    pos = (len(sorted_values) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_values[lo]
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def empirical_halfwidths(
    windows: tuple[BacktestWindow, ...], horizon: int, levels: tuple[float, ...] = (0.8, 0.95)
) -> dict[float, tuple[float, ...]]:
    """Empirical interval half-widths per horizon step.

    Pools rolling-origin absolute errors normalized by `sqrt(step)` (so
    wider-ahead steps are comparable), takes the level quantile, and scales
    it back by `sqrt(step)`. This is the split-conformal idea with the
    backtest residuals as the calibration set: cheap, distribution-free,
    and honest that they are in-sample residuals (not a guarantee).
    """
    if not windows:
        msg = "need at least one backtest window"
        raise ValueError(msg)
    if horizon < 1:
        msg = "horizon must be >= 1"
        raise ValueError(msg)
    for level in levels:
        if not 0.0 < level < 1.0:
            msg = f"level must be in (0, 1), got {level}"
            raise ValueError(msg)
    normalized = sorted(
        abs(f - a) / math.sqrt(step)
        for w in windows
        for step, (a, f) in enumerate(zip(w.actuals, w.forecasts, strict=True), start=1)
    )
    return {
        level: tuple(
            float(_quantile(normalized, level) * math.sqrt(step)) for step in range(1, horizon + 1)
        )
        for level in levels
    }
