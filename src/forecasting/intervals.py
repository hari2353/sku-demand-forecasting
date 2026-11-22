"""Prediction intervals for demand forecasts (cross-pollinated from quant/ML).

Conformal-style intervals from residual magnitudes: the half-width is the
(1-alpha) quantile of absolute residuals, giving marginal coverage without
distributional assumptions. Pure Python, no scipy.
"""
from __future__ import annotations

import math


def _quantile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("values must be non-empty")
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must be in [0, 1]")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def conformal_half_width(residuals: list[float], *, alpha: float = 0.1) -> float:
    """Half-width for a (1-alpha) prediction interval from residuals."""
    if not residuals:
        raise ValueError("residuals must be non-empty")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    magnitudes = [abs(r) for r in residuals]
    return round(_quantile(magnitudes, 1 - alpha), 6)


def prediction_interval(
    point_forecast: float, residuals: list[float], *, alpha: float = 0.1
) -> tuple[float, float]:
    hw = conformal_half_width(residuals, alpha=alpha)
    return (point_forecast - hw, point_forecast + hw)


def interval_coverage(actuals: list[float], lowers: list[float], uppers: list[float]) -> float:
    if not (len(actuals) == len(lowers) == len(uppers)):
        raise ValueError("input lengths must match")
    if not actuals:
        raise ValueError("inputs must be non-empty")
    hits = sum(1 for a, lo, hi in zip(actuals, lowers, uppers, strict=True) if lo <= a <= hi)
    return round(hits / len(actuals), 6)
