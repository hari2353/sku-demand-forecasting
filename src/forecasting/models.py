"""Domain models for SKU-level demand forecasting.

Everything downstream (forecasters, backtests, selection) depends on these
typed frozen models - never the other way around.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class DemandPattern(str, Enum):
    """Categorization used by demand-planning teams (Syntetos/Boylan)."""

    SMOOTH = "smooth"          # regular demand, low variance
    TREND = "trend"            # smooth + strong linear growth/decline
    INTERMITTENT = "intermittent"  # many zero periods, small non-zero demands
    LUMPY = "lumpy"            # many zeros + large unpredictable spikes
    ERRATIC = "erratic"         # non-zero but high variance


@dataclass(frozen=True, slots=True)
class SkuSeries:
    """One SKU's weekly demand history."""

    sku_id: str
    description: str
    pattern: DemandPattern
    history: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.sku_id or not self.sku_id.strip():
            msg = "sku_id must be non-empty"
            raise ValueError(msg)
        if len(self.history) < 10:
            msg = f"need >= 10 observations, got {len(self.history)}"
            raise ValueError(msg)
        if any(v < 0 for v in self.history):
            msg = "demand must be non-negative"
            raise ValueError(msg)

    @property
    def n_periods(self) -> int:
        return len(self.history)

    @property
    def zero_fraction(self) -> float:
        return sum(1 for v in self.history if v == 0) / len(self.history)

    @property
    def cv_squared(self) -> float:
        """Squared coefficient of variation of non-zero demands (ADI/CV taxonomy)."""
        nz = [v for v in self.history if v > 0]
        if len(nz) < 2:
            return 0.0
        mean = sum(nz) / len(nz)
        if mean == 0:
            return 0.0
        var = sum((v - mean) ** 2 for v in nz) / (len(nz) - 1)
        return var / (mean * mean)


@dataclass(frozen=True, slots=True)
class ForecastResult:
    """Point forecasts for `horizon` future periods."""

    forecaster: str
    horizon: int
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.values) != self.horizon:
            msg = f"horizon={self.horizon} but {len(self.values)} values produced"
            raise ValueError(msg)
        if any(v < 0 for v in self.values):
            msg = "forecasts must be non-negative"
            raise ValueError(msg)

    @property
    def total(self) -> float:
        return sum(self.values)

    def rounded(self) -> tuple[int, ...]:
        return tuple(round(v) for v in self.values)


@dataclass(frozen=True, slots=True)
class BacktestWindow:
    """One rolling-origin evaluation window."""

    train_end: int          # index where training data ends (exclusive)
    horizon: int            # number of periods scored
    actuals: tuple[float, ...]
    forecasts: tuple[float, ...]
    forecaster: str

    def __post_init__(self) -> None:
        if len(self.actuals) != len(self.forecasts):
            msg = "actuals and forecasts must align"
            raise ValueError(msg)
        if self.horizon != len(self.actuals):
            msg = "horizon must equal scored periods"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class Scorecard:
    """Error metrics for one forecaster on one window set."""

    forecaster: str
    smape: float
    rmse: float
    bias: float          # mean(forecast - actual); + = over-forecast
    n_windows: int
    mase: float = 0.0    # MAE / in-sample naive MAE; scale-free across SKUs
    rmsse: float = 0.0   # RMSE / in-sample naive RMSE

    def __post_init__(self) -> None:
        if self.n_windows < 1:
            msg = "need at least one window"
            raise ValueError(msg)
        if not 0.0 <= self.smape <= 2.0:
            msg = f"sMAPE out of [0, 2]: {self.smape}"
            raise ValueError(msg)
        if self.mase < 0.0 or self.rmsse < 0.0:
            msg = "scaled errors must be non-negative"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class SkuPlan:
    """The final deliverable: selected model + forecast + its scorecard."""

    sku: SkuSeries
    selected_model: str
    forecast: ForecastResult
    scorecard: Scorecard
    candidates: tuple[Scorecard, ...] = field(default_factory=tuple)
    benchmark_mase: float | None = None   # seasonal-naive MASE; None if unscorable
    beats_benchmark: bool = True          # selected MASE <= benchmark MASE
    interval_80: tuple[tuple[float, float], ...] = ()  # (lower, upper) per step
    interval_95: tuple[tuple[float, float], ...] = ()
