"""Portfolio-level orchestration: plan every SKU, aggregate the summary."""

from __future__ import annotations

import math
from dataclasses import dataclass

from forecasting.backtest import BacktestConfig, select_model
from forecasting.forecasters import Forecaster, candidate_pool_for
from forecasting.models import SkuPlan, SkuSeries


@dataclass(frozen=True, slots=True)
class PortfolioSummary:
    """Headline numbers for the whole portfolio run."""

    n_skus: int
    models_selected: dict[str, int]
    mean_smape: float
    worst_sku: str
    worst_smape: float
    best_sku: str
    best_smape: float
    total_forecast_next_horizon: float

    @property
    def selection_breadth(self) -> int:
        return len(self.models_selected)


@dataclass(frozen=True, slots=True)
class PortfolioResult:
    """Everything a run produced."""

    plans: tuple[SkuPlan, ...]
    summary: PortfolioSummary


class PortfolioPlanner:
    """Runs per-SKU automated selection across the whole portfolio.

    By default each SKU gets the candidate pool appropriate to its demand
    pattern (intermittent/lumpy -> Croston/SBA/AutoETS; smooth/trend ->
    naive/trend/AutoETS), exactly like a demand-planning stack should.
    Pass `pool` explicitly to force a single pool portfolio-wide.
    """

    def __init__(
        self,
        config: BacktestConfig | None = None,
        pool: tuple[Forecaster, ...] | None = None,
    ) -> None:
        self._config = config or BacktestConfig()
        self._pool = pool

    def plan(self, skus: list[SkuSeries]) -> PortfolioResult:
        """Plan every SKU; one un-plannable SKU never kills the portfolio run."""
        plans: list[SkuPlan] = []
        self.skipped: dict[str, str] = {}
        for sku in skus:
            pool = self._pool if self._pool is not None else candidate_pool_for(sku)
            try:
                plans.append(select_model(sku, self._config, pool=pool))
            except ValueError as exc:
                self.skipped[sku.sku_id] = str(exc)
        if not plans and self.skipped:
            first_sku = next(iter(self.skipped))
            msg = f"no SKU could be planned (first: {first_sku}: {self.skipped[first_sku]})"
            raise ValueError(msg) from None
        return PortfolioResult(plans=tuple(plans), summary=_summarize(plans))


def _summarize(plans: list[SkuPlan]) -> PortfolioSummary:
    if not plans:
        msg = "no plans to summarize"
        raise ValueError(msg)
    models: dict[str, int] = {}
    for plan in plans:
        models[plan.selected_model] = models.get(plan.selected_model, 0) + 1
    smapes = {plan.sku.sku_id: plan.scorecard.smape for plan in plans}

    def _by_smape(sku_id: str) -> float:
        return smapes[sku_id]

    best_sku = min(smapes, key=_by_smape)
    worst_sku = max(smapes, key=_by_smape)
    return PortfolioSummary(
        n_skus=len(plans),
        models_selected=models,
        mean_smape=sum(smapes.values()) / len(smapes),
        worst_sku=worst_sku,
        worst_smape=smapes[worst_sku],
        best_sku=best_sku,
        best_smape=smapes[best_sku],
        total_forecast_next_horizon=sum(p.forecast.total for p in plans),
    )


def pattern_mix(plans: tuple[SkuPlan, ...]) -> dict[str, int]:
    """Count of SKUs by demand pattern (portfolio composition view)."""
    mix: dict[str, int] = {}
    for plan in plans:
        key = plan.sku.pattern.value
        mix[key] = mix.get(key, 0) + 1
    return mix


def plan_to_dict(plan: SkuPlan) -> dict:
    """JSON serialization for the CLI summary (dashboard contract)."""
    def finite(value: float) -> float | None:
        return round(value, 4) if math.isfinite(value) else None

    return {
        "sku_id": plan.sku.sku_id,
        "description": plan.sku.description,
        "pattern": plan.sku.pattern.value,
        "selected_model": plan.selected_model,
        "smape": round(plan.scorecard.smape, 4),
        "rmse": round(plan.scorecard.rmse, 2),
        "bias": round(plan.scorecard.bias, 2),
        "mase": finite(plan.scorecard.mase),
        "rmsse": finite(plan.scorecard.rmsse),
        "benchmark_mase": finite(plan.benchmark_mase) if plan.benchmark_mase is not None else None,
        "beats_benchmark": plan.beats_benchmark,
        "n_windows": plan.scorecard.n_windows,
        "forecast_next_horizon": [round(v, 1) for v in plan.forecast.values],
        "forecast_total": round(plan.forecast.total, 1),
        "interval_80": [[round(lo, 1), round(hi, 1)] for lo, hi in plan.interval_80],
        "interval_95": [[round(lo, 1), round(hi, 1)] for lo, hi in plan.interval_95],
        "forecast_total_95": [
            round(sum(lo for lo, _ in plan.interval_95), 1),
            round(sum(hi for _, hi in plan.interval_95), 1),
        ],
        "candidates": [
            {
                "forecaster": c.forecaster,
                "smape": round(c.smape, 4),
                "rmse": round(c.rmse, 2),
                "bias": round(c.bias, 2),
                "mase": finite(c.mase),
                "rmsse": finite(c.rmsse),
            }
            for c in plan.candidates
        ],
    }
