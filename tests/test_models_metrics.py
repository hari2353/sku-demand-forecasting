"""Unit tests for domain models, metrics, and the synthetic generator."""

from __future__ import annotations

import math

import pytest

from forecasting.metrics import bias, rmse, smape
from forecasting.models import DemandPattern, SkuSeries
from forecasting.synthetic import classify, make_portfolio, make_sku


class TestModels:
    def test_sku_series_valid(self) -> None:
        s = SkuSeries("S", "d", DemandPattern.SMOOTH, tuple(range(10, 20)))
        assert s.n_periods == 10

    def test_rejects_short_history(self) -> None:
        with pytest.raises(ValueError, match=">= 10"):
            SkuSeries("S", "d", DemandPattern.SMOOTH, (1.0, 2.0))

    def test_rejects_negative_demand(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            SkuSeries("S", "d", DemandPattern.SMOOTH, (-1.0, *([1.0] * 12)))

    def test_zero_fraction(self) -> None:
        alternating = tuple(0.0 if t % 2 == 0 else 1.0 for t in range(10))
        s = SkuSeries("S", "d", DemandPattern.SMOOTH, alternating)
        assert s.zero_fraction == pytest.approx(0.5)

    def test_forecast_result_alignment(self) -> None:
        from forecasting.models import ForecastResult

        with pytest.raises(ValueError, match="values produced"):
            ForecastResult(forecaster="f", horizon=3, values=(1.0, 2.0))

    def test_forecast_result_non_negative(self) -> None:
        from forecasting.models import ForecastResult

        with pytest.raises(ValueError, match="non-negative"):
            ForecastResult(forecaster="f", horizon=2, values=(-1.0, 2.0))

    def test_scorecard_bounds(self) -> None:
        from forecasting.models import Scorecard

        with pytest.raises(ValueError, match="sMAPE"):
            Scorecard(forecaster="f", smape=2.5, rmse=1.0, bias=0.0, n_windows=1)


class TestMetrics:
    def test_smape_perfect(self) -> None:
        assert smape((10.0, 20.0), (10.0, 20.0)) == 0.0

    def test_smape_both_zero_counts_as_perfect(self) -> None:
        assert smape((0.0, 0.0), (0.0, 0.0)) == 0.0

    def test_smape_symmetric(self) -> None:
        assert smape((100.0,), (150.0,)) == pytest.approx(smape((150.0,), (100.0,)))

    def test_rmse(self) -> None:
        assert rmse((3.0, 4.0), (3.0, 6.0)) == pytest.approx(math.sqrt(2.0))

    def test_bias_sign(self) -> None:
        assert bias((10.0,), (12.0,)) == 2.0  # over-forecast -> positive
        assert bias((10.0,), (8.0,)) == -2.0

    def test_length_mismatch_rejected(self) -> None:
        for fn in (smape, rmse, bias):
            with pytest.raises(ValueError, match="mismatch"):
                fn((1.0, 2.0), (1.0,))


class TestSynthetic:
    def test_deterministic_portfolio(self) -> None:
        a = make_portfolio(n_skus=6, seed=11)
        b = make_portfolio(n_skus=6, seed=11)
        assert a == b

    def test_pattern_rotation_covers_all(self, portfolio) -> None:
        patterns = {s.pattern for s in portfolio}
        assert patterns == {
            DemandPattern.SMOOTH,
            DemandPattern.TREND,
            DemandPattern.INTERMITTENT,
            DemandPattern.LUMPY,
        }

    def test_intermittent_has_zeros(self) -> None:
        sku = make_sku(2, seed=3)  # index 2 -> intermittent
        assert sku.zero_fraction > 0.4

    def test_smooth_has_no_zeros(self) -> None:
        sku = make_sku(0, seed=3)  # index 0 -> smooth
        assert sku.zero_fraction < 0.05

    def test_classify_recovers_intermittent(self) -> None:
        history = tuple(5.0 if t % 5 == 0 else 0.0 for t in range(60))
        assert classify(history) in (DemandPattern.INTERMITTENT, DemandPattern.LUMPY)

    def test_classify_recovers_smooth(self) -> None:
        history = tuple(100.0 + 10.0 * math.sin(t) for t in range(60))
        assert classify(history) in (DemandPattern.SMOOTH, DemandPattern.ERRATIC)

    def test_make_sku_rejects_short(self) -> None:
        with pytest.raises(ValueError, match="n_periods"):
            make_sku(0, n_periods=5)
