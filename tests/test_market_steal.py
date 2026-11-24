"""Market-steal additions (2026-11-18): scale-free accuracy (MASE/RMSSE),
empirical prediction intervals, naive benchmark exception reporting.

Compared products treat these as baseline: StatsForecast-style reports carry
scale-free errors, planners get interval ranges with every forecast, and
selection raises an exception when the winner fails to beat a naive benchmark.
"""

from __future__ import annotations

import math

import pytest

from forecasting.backtest import BacktestConfig, backtest, score_forecaster, select_model
from forecasting.forecasters import SeasonalNaive
from forecasting.metrics import empirical_halfwidths, mase, rmsse
from forecasting.portfolio import PortfolioPlanner, plan_to_dict
from forecasting.synthetic import make_portfolio


# ── scale-free metrics ──────────────────────────────────────────────────────
class TestScaleFreeMetrics:
    INSAMPLE = (10.0, 12.0, 11.0, 14.0, 13.0)  # naive MAE = 7/4 = 1.75
    ACTUAL = (15.0, 16.0, 17.0)

    def test_mase_perfect_forecast_is_zero(self) -> None:
        assert mase(self.ACTUAL, self.ACTUAL, self.INSAMPLE) == 0.0

    def test_mase_is_mae_over_insample_naive_mae(self) -> None:
        forecast = (16.0, 14.0, 19.0)  # MAE = (1+2+2)/3 = 5/3
        assert mase(self.ACTUAL, forecast, self.INSAMPLE) == pytest.approx((5 / 3) / 1.75)

    def test_mase_scales_with_error(self) -> None:
        small = mase(self.ACTUAL, (15.5, 16.5, 17.5), self.INSAMPLE)
        large = mase(self.ACTUAL, (17.0, 18.0, 19.0), self.INSAMPLE)
        assert large == pytest.approx(small * 4)

    def test_rmsse_perfect_forecast_is_zero(self) -> None:
        assert rmsse(self.ACTUAL, self.ACTUAL, self.INSAMPLE) == 0.0

    def test_flat_insample_yields_inf_not_crash(self) -> None:
        flat = (5.0, 5.0, 5.0, 5.0)
        assert mase(self.ACTUAL, (14.0, 15.0, 16.0), flat) == math.inf
        assert rmsse(self.ACTUAL, (14.0, 15.0, 16.0), flat) == math.inf

    def test_length_mismatch_rejected(self) -> None:
        with pytest.raises(ValueError, match="length mismatch"):
            mase(self.ACTUAL, (1.0,), self.INSAMPLE)

    def test_scorecard_carries_mase_and_rmsse(self, smooth_sku) -> None:
        config = BacktestConfig(horizon=13, n_origins=3, min_train=52)
        card = score_forecaster(SeasonalNaive(), smooth_sku, config)
        assert card.mase >= 0.0 and card.rmsse >= 0.0
        assert math.isfinite(card.mase) and math.isfinite(card.rmsse)


# ── empirical prediction intervals ──────────────────────────────────────────
class TestEmpiricalIntervals:
    def _widths(self, sku, level: float):
        config = BacktestConfig(horizon=13, n_origins=4, min_train=52)
        windows = backtest(SeasonalNaive(), sku, config)
        return empirical_halfwidths(windows, config.horizon, levels=(0.8, 0.95))[level]

    def test_band_shape_and_order(self, smooth_sku) -> None:
        widths = self._widths(smooth_sku, 0.95)
        assert len(widths) == 13
        assert all(w >= 0.0 for w in widths)

    def test_95_is_never_narrower_than_80(self, smooth_sku) -> None:
        for w80, w95 in zip(
            self._widths(smooth_sku, 0.8), self._widths(smooth_sku, 0.95), strict=True
        ):
            assert w95 >= w80 - 1e-9

    def test_bands_widen_with_horizon(self, smooth_sku) -> None:
        widths = list(self._widths(smooth_sku, 0.95))
        assert widths == sorted(widths)  # non-decreasing by construction

    def test_forecast_sits_inside_band(self, smooth_sku) -> None:
        config = BacktestConfig(horizon=13, n_origins=4, min_train=52)
        plan = select_model(smooth_sku, config)
        assert len(plan.interval_95) == plan.forecast.horizon
        for value, (lo, hi) in zip(plan.forecast.values, plan.interval_95, strict=True):
            assert lo <= value <= hi

    def test_empty_windows_rejected(self) -> None:
        with pytest.raises(ValueError, match="window"):
            empirical_halfwidths((), 13)


# ── naive benchmark exception reporting ─────────────────────────────────────
class TestBenchmarkException:
    def test_benchmark_mase_matches_standalone_scoring(self, smooth_sku) -> None:
        config = BacktestConfig(horizon=13, n_origins=4, min_train=52)
        plan = select_model(smooth_sku, config)
        expected = score_forecaster(SeasonalNaive(), smooth_sku, config).mase
        assert plan.benchmark_mase == pytest.approx(expected)
        assert plan.beats_benchmark is True  # seasonal naive is in the smooth pool

    def test_selection_unchanged_and_flag_consistent(self) -> None:
        config = BacktestConfig(horizon=13, n_origins=3, min_train=52)
        skus = make_portfolio(n_skus=4, seed=5)
        for sku in skus:
            plan = select_model(sku, config)
            expected = bool(plan.scorecard.mase <= plan.benchmark_mase)
            assert plan.beats_benchmark is expected

    def test_unscorable_benchmark_is_flagged_true(self, monkeypatch) -> None:
        import importlib

        bt = importlib.import_module("forecasting.backtest")
        original = bt.score_forecaster

        def fake(forecaster, sku, config):
            if forecaster.name == "SeasonalNaive(m=52)":
                raise ValueError("benchmark unavailable")
            return original(forecaster, sku, config)

        monkeypatch.setattr(bt, "score_forecaster", fake)
        config = BacktestConfig(horizon=13, n_origins=3, min_train=52)
        plan = select_model(make_portfolio(n_skus=1, seed=5)[0], config)
        assert plan.benchmark_mase is None
        assert plan.beats_benchmark is True

    def test_plan_to_dict_exposes_new_fields(self) -> None:
        planner = PortfolioPlanner(BacktestConfig(horizon=13, n_origins=2))
        result = planner.plan(make_portfolio(n_skus=2, seed=5))
        d = plan_to_dict(result.plans[0])
        for key in (
            "mase",
            "rmsse",
            "benchmark_mase",
            "beats_benchmark",
            "interval_80",
            "interval_95",
            "forecast_total_95",
        ):
            assert key in d
        assert len(d["interval_95"]) == len(d["forecast_next_horizon"])
        assert d["forecast_total_95"][0] <= d["forecast_total_95"][1]
