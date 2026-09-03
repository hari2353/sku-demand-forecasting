"""Forecasters: a Forecaster protocol + five implementations.

All are pure-numpy, deterministic, and stateless fit->forecast pairs:
seasonal naive and linear trend are baselines every serious forecast
stack must beat; AutoETS is a compact exponential-smoothing grid search
(what my Altria tool ran per SKU); Croston/SBA handle intermittent
demand where plain exponential smoothing fails.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from forecasting.models import DemandPattern, ForecastResult, SkuSeries

FloatArray = NDArray[np.float64]


class Forecaster(Protocol):
    """Anything that fits on history and forecasts `horizon` periods."""

    @property
    def name(self) -> str: ...

    def fit(self, history: tuple[float, ...]) -> Forecaster: ...

    def forecast(self, horizon: int) -> ForecastResult: ...


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------


@dataclass
class SeasonalNaive:
    """Last season repeated: the honest baseline for seasonal demand."""

    period: int = 52

    @property
    def name(self) -> str:
        return f"SeasonalNaive(m={self.period})"

    def fit(self, history: tuple[float, ...]) -> SeasonalNaive:
        if len(history) < self.period:
            msg = f"history shorter than season {self.period}"
            raise ValueError(msg)
        self._history = np.asarray(history, dtype=np.float64)
        return self

    def forecast(self, horizon: int) -> ForecastResult:
        h = np.asarray(getattr(self, "_history", ()), dtype=np.float64)
        if h.size == 0:
            msg = "call fit() first"
            raise ValueError(msg)
        if horizon < 1:
            msg = "horizon must be >= 1"
            raise ValueError(msg)
        last_season = h[-self.period :]
        reps = int(np.ceil(horizon / self.period))
        tiled = np.tile(last_season, reps)[:horizon]
        return ForecastResult(
            forecaster=self.name, horizon=horizon, values=tuple(tiled)
        )


@dataclass
class LinearTrend:
    """OLS linear trend + mean-reverting noise clamp."""

    @property
    def name(self) -> str:
        return "LinearTrend"

    def fit(self, history: tuple[float, ...]) -> LinearTrend:
        if len(history) < 3:
            msg = "need >= 3 observations for a trend"
            raise ValueError(msg)
        y = np.asarray(history, dtype=np.float64)
        t = np.arange(len(y), dtype=np.float64)
        slope, intercept = np.polyfit(t, y, 1)
        self._slope = float(slope)
        self._intercept = float(intercept)
        self._last_t = len(y) - 1
        return self

    def forecast(self, horizon: int) -> ForecastResult:
        if not hasattr(self, "_slope"):
            msg = "call fit() first"
            raise ValueError(msg)
        if horizon < 1:
            msg = "horizon must be >= 1"
            raise ValueError(msg)
        future_t = np.arange(self._last_t + 1, self._last_t + 1 + horizon)
        values = np.maximum(0.0, self._intercept + self._slope * future_t)
        return ForecastResult(
            forecaster=self.name, horizon=horizon, values=tuple(values)
        )


# ---------------------------------------------------------------------------
# AutoETS (grid-searched exponential smoothing)
# ---------------------------------------------------------------------------


def _ets_forecast(
    y: FloatArray, horizon: int, *, alpha: float, beta: float, phi: float
) -> FloatArray:
    """Holt damped-trend exponential smoothing, numpy-only."""
    level = float(y[0])
    trend = 0.0
    for v in y[1:]:
        prev_level = level
        level = alpha * float(v) + (1 - alpha) * (level + phi * trend)
        trend = beta * (level - prev_level) + (1 - beta) * phi * trend
    out = []
    for _ in range(horizon):
        out.append(max(0.0, level + trend))
        trend *= phi  # damped: trend decays toward zero
    return np.asarray(out, dtype=np.float64)


@dataclass
class AutoETS:
    """Compact ETS grid: alpha x beta x phi, selected by in-sample one-step
    squared error (mirrors the auto-selection step my Altria tool performed)."""

    alphas: tuple[float, ...] = (0.05, 0.15, 0.30, 0.50, 0.80)
    betas: tuple[float, ...] = (0.0, 0.10, 0.30)
    phis: tuple[float, ...] = (0.85, 0.95, 1.0)

    @property
    def name(self) -> str:
        return "AutoETS"

    def fit(self, history: tuple[float, ...]) -> AutoETS:
        if len(history) < 4:
            msg = "need >= 4 observations for ETS"
            raise ValueError(msg)
        y = np.asarray(history, dtype=np.float64)
        best_params: tuple[float, float, float] | None = None
        best_sse = float("inf")
        for alpha in self.alphas:
            for beta in self.betas:
                for phi in self.phis:
                    sse = self._one_step_sse(y, alpha, beta, phi)
                    if sse < best_sse:
                        best_sse = sse
                        best_params = (alpha, beta, phi)
        assert best_params is not None
        self._best = best_params
        self._y = y
        return self

    @staticmethod
    def _one_step_sse(y: FloatArray, alpha: float, beta: float, phi: float) -> float:
        level = float(y[0])
        trend = 0.0
        sse = 0.0
        for v in y[1:]:
            pred = level + trend
            sse += (float(v) - pred) ** 2
            prev_level = level
            level = alpha * float(v) + (1 - alpha) * (level + phi * trend)
            trend = beta * (level - prev_level) + (1 - beta) * phi * trend
        return sse

    def forecast(self, horizon: int) -> ForecastResult:
        if not hasattr(self, "_best"):
            msg = "call fit() first"
            raise ValueError(msg)
        if horizon < 1:
            msg = "horizon must be >= 1"
            raise ValueError(msg)
        alpha, beta, phi = self._best
        values = _ets_forecast(self._y, horizon, alpha=alpha, beta=beta, phi=phi)
        return ForecastResult(
            forecaster=self.name, horizon=horizon, values=tuple(values)
        )


# ---------------------------------------------------------------------------
# Intermittent-demand methods
# ---------------------------------------------------------------------------


@dataclass
class Croston:
    """Classic Croston (1972): separate exponential smoothing on demand
    size and inter-arrival interval; forecast = size / interval."""

    alpha: float = 0.1

    @property
    def name(self) -> str:
        return "Croston"

    def fit(self, history: tuple[float, ...]) -> Croston:
        if not any(v > 0 for v in history):
            msg = "Croston needs at least one non-zero demand"
            raise ValueError(msg)
        self._z, self._p = _croston_states(history, alpha=self.alpha)
        return self

    def forecast(self, horizon: int) -> ForecastResult:
        if not hasattr(self, "_z"):
            msg = "call fit() first"
            raise ValueError(msg)
        if horizon < 1:
            msg = "horizon must be >= 1"
            raise ValueError(msg)
        per_period = max(0.0, self._z / self._p) if self._p > 0 else 0.0
        values = tuple(per_period for _ in range(horizon))
        return ForecastResult(
            forecaster=self.name, horizon=horizon, values=values
        )


@dataclass
class SBA:
    """Syntetos-Boylan approximation: Croston with debias factor
    (1 - alpha/2) - the standard correction for Croston's positive bias."""

    alpha: float = 0.1

    @property
    def name(self) -> str:
        return "SBA"

    def fit(self, history: tuple[float, ...]) -> SBA:
        if not any(v > 0 for v in history):
            msg = "SBA needs at least one non-zero demand"
            raise ValueError(msg)
        z, p = _croston_states(history, alpha=self.alpha)
        self._z, self._p = z, p
        return self

    def forecast(self, horizon: int) -> ForecastResult:
        if not hasattr(self, "_z"):
            msg = "call fit() first"
            raise ValueError(msg)
        if horizon < 1:
            msg = "horizon must be >= 1"
            raise ValueError(msg)
        per_period = max(0.0, (1.0 - self.alpha / 2.0) * self._z / self._p) \
            if self._p > 0 else 0.0
        values = tuple(per_period for _ in range(horizon))
        return ForecastResult(
            forecaster=self.name, horizon=horizon, values=values
        )


def _croston_states(
    history: tuple[float, ...], *, alpha: float
) -> tuple[float, float]:
    """Smoothed demand size (z) and interval (p) after Croston (1972)."""
    z = 0.0
    p = 0.0
    since_last = 0
    seen_demand = False
    for v in history:
        if v > 0:
            if not seen_demand:
                z = v
                p = max(1.0, float(since_last)) if since_last > 0 else 1.0
                seen_demand = True
            else:
                z = z + alpha * (v - z)
                p = p + alpha * (since_last - p)
            since_last = 0
        else:
            since_last += 1
    if not seen_demand:
        msg = "no non-zero demand"
        raise ValueError(msg)
    return z, max(p, 1.0)


DEFAULT_FORECASTERS: tuple[Forecaster, ...] = (
    SeasonalNaive(),
    LinearTrend(),
    AutoETS(),
    Croston(),
    SBA(),
)


def candidate_pool_for(sku: SkuSeries) -> tuple[Forecaster, ...]:
    """Sensible candidate set by demand pattern (small, interpretable grids)."""
    if sku.pattern in (DemandPattern.INTERMITTENT, DemandPattern.LUMPY):
        return (Croston(), SBA(), AutoETS())
    return (SeasonalNaive(), LinearTrend(), AutoETS())


def safe_forecast(forecaster: Forecaster, history: tuple[float, ...], horizon: int):
    """Forecast with graceful degradation: degenerate series fall back to
    the historical mean so one bad SKU never kills a portfolio run."""
    from forecasting.models import ForecastResult

    try:
        return forecaster.fit(history).forecast(horizon)
    except ValueError:
        mean = max(0.0, sum(history) / len(history)) if history else 0.0
        return ForecastResult(
            forecaster=f"{forecaster.name}+mean_fallback",
            horizon=horizon,
            values=tuple(mean for _ in range(horizon)),
        )
