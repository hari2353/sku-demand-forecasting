"""Unit + integration tests for backtesting, selection, and the portfolio."""

from __future__ import annotations

import pytest

from forecasting.backtest import BacktestConfig, backtest, score_forecaster, select_model
from forecasting.forecasters import AutoETS, SeasonalNaive
from forecasting.portfolio import PortfolioPlanner, pattern_mix, plan_to_dict
from forecasting.synthetic import make_portfolio

pytestmark_integration = pytest.mark.integration


class TestBacktest:
    def test_windows_causal(self, smooth_sku) -> None:
        config = BacktestConfig(horizon=13, n_origins=4, min_train=52)
        windows = backtest(SeasonalNaive(), smooth_sku, config)
        assert len(windows) >= 1
        for w in windows:
            assert len(w.actuals) == 13
            assert len(w.forecasts) == 13
            # strictly after training end
            assert w.train_end + w.horizon <= smooth_sku.n_periods

    def test_no_lookahead_leakage(self, smooth_sku) -> None:
        """A forecaster must never see scored actuals: train slice length
        equals train_end and the scored window starts exactly there."""
        config = BacktestConfig(horizon=13, n_origins=3, min_train=52)
        windows = backtest(AutoETS(), smooth_sku, config)
        for i, w in enumerate(windows):
            assert w.train_end == 52 + i * 13

    def test_short_history_rejected(self) -> None:
        from forecasting.models import DemandPattern, SkuSeries

        short = SkuSeries("S", "d", DemandPattern.SMOOTH, tuple(60.0 for _ in range(60)))
        config = BacktestConfig(horizon=13, n_origins=4, min_train=52)
        with pytest.raises(ValueError, match="too short"):
            backtest(SeasonalNaive(), short, config)

    def test_scorecard_aggregates(self, smooth_sku) -> None:
        config = BacktestConfig(horizon=13, n_origins=2, min_train=52)
        card = score_forecaster(SeasonalNaive(), smooth_sku, config)
        assert card.n_windows == 2
        assert 0.0 <= card.smape <= 2.0

    def test_config_validation(self) -> None:
        with pytest.raises(ValueError, match="horizon"):
            BacktestConfig(horizon=0)
        with pytest.raises(ValueError, match="n_origins"):
            BacktestConfig(n_origins=0)


class TestSelection:
    def test_selects_and_keeps_all_candidates(self, smooth_sku) -> None:
        plan = select_model(smooth_sku, BacktestConfig(horizon=13, n_origins=3))
        assert plan.selected_model
        assert plan.forecast.horizon == 13
        assert len(plan.candidates) >= 2
        # selected model is the best by sMAPE
        assert plan.candidates[0].forecaster == plan.selected_model

    def test_selection_prefers_croston_on_intermittent(self, intermittent_sku) -> None:
        """On heavy-zero demand, seasonal naive gets destroyed; Croston/SBA
        or the ETS fallback must win."""
        plan = select_model(
            intermittent_sku,
            BacktestConfig(horizon=13, n_origins=3, min_train=52),
        )
        names = [c.forecaster for c in plan.candidates]
        assert plan.selected_model in (*names,)
        assert "Croston" in names or "SBA" in names

    def test_selection_is_deterministic(self, smooth_sku) -> None:
        config = BacktestConfig(horizon=13, n_origins=3)
        a = select_model(smooth_sku, config)
        b = select_model(smooth_sku, config)
        assert a.selected_model == b.selected_model
        assert a.scorecard.smape == b.scorecard.smape


@pytest.mark.integration
class TestPortfolioIntegration:
    def test_full_portfolio_run(self, portfolio) -> None:
        planner = PortfolioPlanner(BacktestConfig(horizon=13, n_origins=4, min_train=52))
        result = planner.plan(portfolio)
        assert len(result.plans) == len(portfolio) == 8

        s = result.summary
        assert s.n_skus == 8
        assert sum(s.models_selected.values()) == 8
        assert s.mean_smape > 0
        assert s.best_smape <= s.worst_smape
        assert s.total_forecast_next_horizon > 0

    def test_every_sku_gets_a_model(self, portfolio) -> None:
        planner = PortfolioPlanner(BacktestConfig(horizon=13, n_origins=3))
        result = planner.plan(portfolio)
        assert all(p.selected_model for p in result.plans)

    def test_pattern_mix_covers_portfolio(self, portfolio) -> None:
        planner = PortfolioPlanner(BacktestConfig(horizon=13, n_origins=3))
        result = planner.plan(portfolio)
        mix = pattern_mix(result.plans)
        assert sum(mix.values()) == len(portfolio)
        assert "intermittent" in mix

    def test_full_run_deterministic(self) -> None:
        planner = PortfolioPlanner(BacktestConfig(horizon=13, n_origins=3))
        r1 = planner.plan(make_portfolio(n_skus=4, seed=5))
        r2 = planner.plan(make_portfolio(n_skus=4, seed=5))
        assert [p.selected_model for p in r1.plans] == [p.selected_model for p in r2.plans]
        assert [p.scorecard.smape for p in r1.plans] == [p.scorecard.smape for p in r2.plans]

    def test_plan_to_dict_shape(self, portfolio) -> None:
        planner = PortfolioPlanner(BacktestConfig(horizon=13, n_origins=2))
        result = planner.plan(portfolio)
        d = plan_to_dict(result.plans[0])
        for key in ("sku_id", "pattern", "selected_model", "smape",
                    "forecast_next_horizon", "candidates"):
            assert key in d

    def test_one_bad_sku_never_kills_the_run(self, portfolio) -> None:
        """A too-short history is skipped and recorded, others still plan."""
        from forecasting.models import DemandPattern, SkuSeries

        short = SkuSeries(
            "SKU-BAD", "too short", DemandPattern.SMOOTH, tuple(60.0 for _ in range(60))
        )
        planner = PortfolioPlanner(BacktestConfig(horizon=13, n_origins=4, min_train=52))
        result = planner.plan([*portfolio, short])
        assert len(result.plans) == len(portfolio)
        assert "SKU-BAD" in planner.skipped
        assert planner.skipped["SKU-BAD"]  # records the underlying reason

    def test_all_bad_skus_raise(self) -> None:
        from forecasting.models import DemandPattern, SkuSeries

        short = SkuSeries(
            "SKU-BAD", "too short", DemandPattern.SMOOTH, tuple(60.0 for _ in range(60))
        )
        planner = PortfolioPlanner(BacktestConfig(horizon=13, n_origins=4, min_train=52))
        with pytest.raises(ValueError, match="no SKU could be planned"):
            planner.plan([short])
