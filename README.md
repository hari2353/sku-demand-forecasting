# SKU Demand Forecasting with Automated Model Selection

**Public portfolio replica of my Altria demand-forecasting tool** — weekly/monthly SKU-level sales prediction with automated per-SKU model selection: rolling-origin backtests score every candidate, the best model wins per SKU.

The demo portfolio covers all four demand archetypes a real planner faces, and the selection table shows the models earning their keep:

```
$ python scripts/run_forecast.py --out artifacts

SKU            pattern       selected model   sMAPE     RMSE    bias next-Q total
------------------------------------------------------------------------
SKU-0420009    trend         AutoETS          0.040      6.4   -2.79       2157.6
SKU-0420001    trend         AutoETS          0.045      5.4   -2.47       1680.2
SKU-0420005    trend         AutoETS          0.054      9.0   -4.82       2248.2
SKU-0420000    smooth        SeasonalNaive(m=52)   0.060      6.9    2.29       1436.5
SKU-0420008    smooth        SeasonalNaive(m=52)   0.061     10.4    0.87       2053.0
SKU-0420004    smooth        SeasonalNaive(m=52)   0.062      9.6    0.96       1993.1
SKU-0420010    intermittent  Croston          1.375      2.6    1.43         24.6
SKU-0420006    intermittent  Croston          1.386      2.5    1.28         38.9
SKU-0420002    intermittent  Croston          1.547      2.9    1.58         26.0
SKU-0420003    lumpy         Croston          1.618     10.8   -0.02        138.7
SKU-0420007    lumpy         Croston          1.670      5.6    3.20        147.1
SKU-0420011    lumpy         Croston          1.717     16.3    1.22         87.4
------------------------------------------------------------------------
models selected : SeasonalNaive 3 | AutoETS 3 | Croston 6
portfolio next-quarter total: 12,031 units
```

Read the table: smooth SKUs won by seasonal naive (hard to beat on stable seasonal demand), trend SKUs needed **AutoETS** to track growth, and every intermittent/lumpy SKU went to **Croston** — exactly the methods-intuition split a demand-planning interviewer wants to see. (High sMAPE on intermittent SKUs is the honest reality of sparse demand; RMSE stays small because the volumes are small.)

**Verified quality bar:** 53 tests (unit + forecaster-contract + portfolio integration), 91% coverage, ruff + mypy clean, CI on Python 3.10-3.12 with a full portfolio run.

## Models included

| Model | Designed for | How |
|---|---|---|
| `SeasonalNaive` | smooth seasonal demand | last 52-week season repeated; the baseline to beat |
| `LinearTrend` | growth/decline SKUs | OLS trend extrapolation, clamped at zero |
| `AutoETS` | trend/seasonal | grid-searched Holt damped-trend exponential smoothing (alpha x beta x phi), picked by one-step in-sample SSE |
| `Croston` | intermittent demand | separate smoothing of demand size and inter-arrival interval |
| `SBA` | intermittent demand | Croston with the Syntetos-Boylan debias factor `(1 - alpha/2)` |

Candidate pools are **pattern-aware by default**: intermittent/lumpy SKUs compete Croston/SBA/AutoETS; smooth/trend SKUs compete naive/trend/AutoETS (the ADI/CV² taxonomy from Syntetos & Boylan drives this).

## Evaluation you can trust

- **Rolling-origin backtests** — strictly causal: every fit sees only data before the window it scores; 4 origins x 13-week horizon by default
- **sMAPE** (zero-robust, the right headline metric for sparse demand), **RMSE**, and **signed bias** (positive = over-forecasting — the number planners actually argue about)
- **Graceful degradation** — one bad SKU is recorded and skipped, never kills the portfolio run
- **Deterministic everywhere** — same seed, same portfolio, same selections, on every machine and every CI run

## Design (SOLID)

- `Forecaster` is a **protocol**; five implementations, all stateless fit→forecast pairs — adding XGBoost/Prophet/ARIMA = one new class, zero pipeline edits (OCP)
- `PortfolioPlanner`, `select_model`, `backtest` are pure functions of their inputs — trivially testable, no hidden state
- The selection criterion and backtest config live in `BacktestConfig`/`Scorecard` — swap metrics without touching forecaster code
- `safe_forecast` wraps every candidate: degenerate histories fall back to the historical mean, keeping the pipeline robust to real-world dirty data

## Quickstart

```bash
pip install -e .[dev]
pytest                                  # 53 tests
python scripts/run_forecast.py          # the table above
python scripts/run_forecast.py --skus 24 --seed 7   # bigger portfolio
```

## Repository layout

```
src/forecasting/
  models.py       # typed frozen domain: series, forecasts, windows, scorecards
  metrics.py      # sMAPE / RMSE / bias (single source of truth)
  synthetic.py    # deterministic 4-archetype portfolio + ADI/CV2 classifier
  forecasters.py  # Forecaster protocol + 5 models + pattern-aware pools
  backtest.py     # rolling-origin evaluation + per-SKU selection
  portfolio.py    # portfolio orchestration + summary + JSON export
scripts/
  run_forecast.py   # the demo run
tests/
  test_models_metrics.py    # domain + metrics
  test_forecasters.py       # contract + per-model behavior
  test_backtest_portfolio.py # causality, selection, end-to-end portfolio
```

## Why synthetic data?

The real tool ran on Altria's proprietary shipment histories. The *machinery* — backtesting discipline, pattern-aware candidate pools, automated selection, bias tracking — is fully demonstrated here on data anyone can regenerate with one seed.

## License

MIT
