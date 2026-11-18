"""Rolling-origin backtesting and per-SKU model selection.

The backtest walks the history forward one origin at a time (no leakage:
each fit only sees data before the horizon being scored). Selection is a
pure function: score every candidate pool, pick lowest sMAPE (RMSE as
tiebreak) - exactly the automated-selection framework of my Altria tool.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from forecasting.forecasters import (
    Forecaster,
    SeasonalNaive,
    candidate_pool_for,
    safe_forecast,
)
from forecasting.metrics import bias, empirical_halfwidths, rmse, smape
from forecasting.models import (
    BacktestWindow,
    ForecastResult,
    Scorecard,
    SkuPlan,
    SkuSeries,
)


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    """How rolling-origin evaluation is run."""

    horizon: int = 13          # one quarter of weekly buckets
    n_origins: int = 4        # how many rolling origins to score
    min_train: int = 52       # minimum training length before first origin

    def __post_init__(self) -> None:
        if self.horizon < 1:
            msg = "horizon must be >= 1"
            raise ValueError(msg)
        if self.n_origins < 1:
            msg = "n_origins must be >= 1"
            raise ValueError(msg)
        if self.min_train < 10:
            msg = "min_train must be >= 10"
            raise ValueError(msg)


def backtest(
    forecaster: Forecaster,
    sku: SkuSeries,
    config: BacktestConfig,
) -> tuple[BacktestWindow, ...]:
    """Rolling-origin windows for one forecaster; strictly causal fits."""
    h = sku.history
    if len(h) < config.min_train + config.horizon * config.n_origins:
        msg = (
            f"{sku.sku_id}: history too short for "
            f"{config.n_origins} origins of {config.horizon} (need "
            f"{config.min_train + config.horizon * config.n_origins}, got {len(h)})"
        )
        raise ValueError(msg)

    windows: list[BacktestWindow] = []
    for origin in range(config.n_origins):
        train_end = config.min_train + origin * config.horizon
        test_end = train_end + config.horizon
        if test_end > len(h):
            break
        train = h[:train_end]
        actuals = h[train_end:test_end]
        result = safe_forecast(forecaster, train, config.horizon)
        windows.append(
            BacktestWindow(
                train_end=train_end,
                horizon=config.horizon,
                actuals=tuple(actuals),
                forecasts=tuple(result.values),
                forecaster=forecaster.name,
            )
        )
    if not windows:
        msg = f"{sku.sku_id}: no backtest windows produced"
        raise ValueError(msg)
    return tuple(windows)


def score_forecaster(
    forecaster: Forecaster,
    sku: SkuSeries,
    config: BacktestConfig,
) -> Scorecard:
    """Aggregate one forecaster's rolling-origin windows into a scorecard."""
    windows = backtest(forecaster, sku, config)
    all_actuals: tuple[float, ...] = ()
    all_forecasts: tuple[float, ...] = ()
    naive_abs = 0.0
    naive_sq = 0.0
    naive_points = 0
    for w in windows:
        all_actuals = all_actuals + w.actuals
        all_forecasts = all_forecasts + w.forecasts
        train = sku.history[: w.train_end]
        naive_abs += sum(abs(train[i] - train[i - 1]) for i in range(1, len(train)))
        naive_sq += sum((train[i] - train[i - 1]) ** 2 for i in range(1, len(train)))
        naive_points += len(train) - 1
    abs_err = sum(abs(f - a) for a, f in zip(all_actuals, all_forecasts, strict=True))
    sq_err = sum((f - a) ** 2 for a, f in zip(all_actuals, all_forecasts, strict=True))
    mean_abs_err = abs_err / len(all_actuals)
    mean_sq_err = sq_err / len(all_actuals)
    mean_naive_abs = naive_abs / naive_points if naive_points else 0.0
    mean_naive_sq = naive_sq / naive_points if naive_points else 0.0
    return Scorecard(
        forecaster=forecaster.name,
        smape=smape(all_actuals, all_forecasts),
        rmse=rmse(all_actuals, all_forecasts),
        bias=bias(all_actuals, all_forecasts),
        n_windows=len(windows),
        # mean error / mean in-sample naive error (standard MASE/RMSSE scaling)
        mase=math.inf if mean_naive_abs == 0 else mean_abs_err / mean_naive_abs,
        rmsse=math.inf if mean_naive_sq == 0 else math.sqrt(mean_sq_err / mean_naive_sq),
    )


def select_model(
    sku: SkuSeries,
    config: BacktestConfig,
    *,
    pool: tuple[Forecaster, ...] | None = None,
) -> SkuPlan:
    """Score every candidate, pick the best, and keep the full comparison."""
    candidates = pool if pool is not None else candidate_pool_for(sku)
    cards: list[Scorecard] = []
    for forecaster in candidates:
        try:
            cards.append(score_forecaster(forecaster, sku, config))
        except ValueError:
            continue  # forecaster incompatible with this SKU's history
    if not cards:
        msg = f"{sku.sku_id}: no candidate could be scored"
        raise ValueError(msg)

    ranked = sorted(cards, key=lambda c: (c.smape, c.rmse))
    best = ranked[0]
    winner = next(f for f in candidates if f.name == best.forecaster)

    final_forecast: ForecastResult = safe_forecast(
        winner, sku.history, config.horizon
    )
    # Benchmark exception: score seasonal naive regardless of the SKU's pool.
    try:
        benchmark = score_forecaster(SeasonalNaive(), sku, config)
    except ValueError:
        benchmark = None
    windows = backtest(winner, sku, config)
    bands = empirical_halfwidths(windows, config.horizon)
    point = final_forecast.values
    return SkuPlan(
        sku=sku,
        selected_model=best.forecaster,
        forecast=final_forecast,
        scorecard=best,
        candidates=tuple(sorted(cards, key=lambda c: c.smape)),
        benchmark_mase=benchmark.mase if benchmark is not None else None,
        beats_benchmark=bool(benchmark is None or best.mase <= benchmark.mase),
        interval_80=tuple(
            (max(0.0, v - hw), v + hw)
            for v, hw in zip(point, bands[0.8], strict=True)
        ),
        interval_95=tuple(
            (max(0.0, v - hw), v + hw)
            for v, hw in zip(point, bands[0.95], strict=True)
        ),
    )
