"""Run automated per-SKU model selection over the synthetic portfolio.

Story told by this script (all numbers reproducible via --seed):

1. build a deterministic 12-SKU portfolio covering smooth/trend/
   intermittent/lumpy demand;
2. rolling-origin backtest every candidate forecaster per SKU;
3. auto-select the best model per SKU (sMAPE, RMSE tiebreak);
4. print the selection table, pattern mix, and next-quarter totals.

Usage:
    python scripts/run_forecast.py --out artifacts
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from forecasting.backtest import BacktestConfig
from forecasting.portfolio import (
    PortfolioPlanner,
    pattern_mix,
    plan_to_dict,
)
from forecasting.synthetic import make_portfolio


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("artifacts"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skus", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=13)
    args = parser.parse_args()

    print("=" * 72)
    print(f"SKU DEMAND FORECASTING - automated model selection (seed={args.seed})")
    print("=" * 72)

    skus = make_portfolio(n_skus=args.skus, seed=args.seed)
    print(f"portfolio: {len(skus)} SKUs, {skus[0].n_periods} weekly buckets each")
    print()

    config = BacktestConfig(horizon=args.horizon, n_origins=4, min_train=52)
    planner = PortfolioPlanner(config=config)
    result = planner.plan(skus)
    if planner.skipped:
        print(f"skipped SKUs     : {len(planner.skipped)} (e.g. "
              f"{next(iter(planner.skipped))}: {next(iter(planner.skipped.values()))})")
        print()

    print(f"{'SKU':<14} {'pattern':<13} {'selected model':<14} "
          f"{'sMAPE':>7} {'MASE':>6} {'RMSE':>8} {'bias':>7} {'next-Q total':>12}")
    print("-" * 80)
    for plan in sorted(result.plans, key=lambda p: p.scorecard.smape):
        print(
            f"{plan.sku.sku_id:<14} {plan.sku.pattern.value:<13} "
            f"{plan.selected_model:<14} {plan.scorecard.smape:>7.3f} "
            f"{plan.scorecard.mase:>6.2f} "
            f"{plan.scorecard.rmse:>8.1f} {plan.scorecard.bias:>7.2f} "
            f"{plan.forecast.total:>12.1f}"
        )

    s = result.summary
    print("-" * 80)
    print(f"models selected  : {s.models_selected}")
    print(f"pattern mix      : {pattern_mix(result.plans)}")
    print(f"mean sMAPE       : {s.mean_smape:.3f}")
    print(f"best SKU         : {s.best_sku} ({s.best_smape:.3f})")
    print(f"worst SKU        : {s.worst_sku} ({s.worst_smape:.3f})")
    low_95 = sum(lo for plan in result.plans for lo, _ in plan.interval_95)
    high_95 = sum(hi for plan in result.plans for _, hi in plan.interval_95)
    print(f"portfolio next-Q : {s.total_forecast_next_horizon:.0f} units "
          f"(95% range {low_95:.0f}-{high_95:.0f})")
    exceptions = [p for p in result.plans if not p.beats_benchmark]
    if exceptions:
        print(f"benchmark exceptions: {len(exceptions)} SKU(s) worse than seasonal naive on MASE")
        for plan in exceptions:
            print(f"  {plan.sku.sku_id}: {plan.selected_model} MASE "
                  f"{plan.scorecard.mase:.2f} vs naive {plan.benchmark_mase:.2f}")
    else:
        print("benchmark exceptions: none (every winner beats seasonal naive on MASE)")

    args.out.mkdir(parents=True, exist_ok=True)
    summary = {
        "seed": args.seed,
        "overall": {
            "n_skus": s.n_skus,
            "models_selected": s.models_selected,
            "smape": round(s.mean_smape, 4),
            "best_sku": s.best_sku,
            "worst_sku": s.worst_sku,
            "total_forecast_next_horizon": round(s.total_forecast_next_horizon, 1),
            "total_forecast_95_range": [round(low_95, 1), round(high_95, 1)],
            "benchmark_exceptions": [p.sku.sku_id for p in exceptions],
        },
        "per_sku": {plan.sku.sku_id: plan_to_dict(plan) for plan in result.plans},
    }
    out_file = args.out / "portfolio_summary.json"
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print()
    print(f"summary written to {out_file}")


if __name__ == "__main__":
    main()
