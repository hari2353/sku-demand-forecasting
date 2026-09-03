"""Unit tests for every forecaster + the shared Forecaster contract."""

from __future__ import annotations

import pytest

from forecasting.forecasters import (
    SBA,
    AutoETS,
    Croston,
    LinearTrend,
    SeasonalNaive,
    candidate_pool_for,
    safe_forecast,
)
from forecasting.models import ForecastResult


def _fresh_forecasters() -> list:
    """New instances per call: forecasters are mutable fit-state objects."""
    return [SeasonalNaive(), LinearTrend(), AutoETS(), Croston(), SBA()]


class TestForecasterContract:
    """Every Forecaster must satisfy: fit -> forecast -> aligned, non-negative."""

    def test_contract_all_forecasters(self, smooth_sku) -> None:
        for f in _fresh_forecasters():
            result = f.fit(smooth_sku.history).forecast(13)
            assert isinstance(result, ForecastResult)
            assert result.horizon == 13
            assert len(result.values) == 13
            assert all(v >= 0 for v in result.values)
            assert result.forecaster == f.name

    def test_forecast_before_fit_raises(self) -> None:
        for f in _fresh_forecasters():
            with pytest.raises(ValueError, match="fit"):
                f.forecast(4)

    def test_horizon_validation(self, smooth_sku) -> None:
        for f in _fresh_forecasters():
            fitted = f.fit(smooth_sku.history)
            with pytest.raises(ValueError, match="horizon"):
                fitted.forecast(0)


class TestSeasonalNaive:
    def test_repeats_last_season(self) -> None:
        history = tuple(float(i % 4) for i in range(16))  # season 4
        result = SeasonalNaive(period=4).fit(history).forecast(8)
        assert result.values == (0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0)

    def test_short_history_rejected(self) -> None:
        with pytest.raises(ValueError, match="shorter than season"):
            SeasonalNaive(period=52).fit((1.0,) * 10)


class TestLinearTrend:
    def test_fits_growing_series(self) -> None:
        history = tuple(float(t) * 2.0 for t in range(20))
        result = LinearTrend().fit(history).forecast(5)
        # extrapolates the upward line
        assert result.values[0] < result.values[-1]
        assert result.values[0] > 36.0

    def test_flat_series_forecasts_flat(self) -> None:
        result = LinearTrend().fit((50.0,) * 30).forecast(5)
        assert all(v == pytest.approx(50.0) for v in result.values)


class TestAutoETS:
    def test_smooth_series_forecast_near_level(self, smooth_sku) -> None:
        result = AutoETS().fit(smooth_sku.history).forecast(13)
        mean = sum(smooth_sku.history[-13:]) / 13
        # forecasts should stay within a sane band of the recent mean
        assert all(abs(v - mean) < 0.5 * mean for v in result.values)

    def test_damped_trend_never_explodes(self) -> None:
        history = tuple(float(t) for t in range(30))
        result = AutoETS().fit(history).forecast(50)
        assert result.values[-1] <= max(history) * 3.0


class TestCrostonSBA:
    def test_croston_pools_demand(self) -> None:
        # demand every 5th period of size 10 -> rate ~2/period
        history = tuple(10.0 if t % 5 == 0 else 0.0 for t in range(60))
        result = Croston(alpha=0.1).fit(history).forecast(13)
        assert all(0 < v < 5.0 for v in result.values)

    def test_sba_below_croston(self) -> None:
        """SBA debiases Croston's positive bias -> equal or lower forecast."""
        history = tuple(8.0 if t % 4 == 0 else 0.0 for t in range(80))
        croston = Croston(alpha=0.1).fit(history).forecast(1).values[0]
        sba = SBA(alpha=0.1).fit(history).forecast(1).values[0]
        assert sba <= croston

    def test_all_zero_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-zero"):
            Croston().fit((0.0,) * 20)

    def test_constant_rate_series(self) -> None:
        history = (1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0)
        result = Croston(alpha=0.2).fit(history).forecast(4)
        assert all(v > 0 for v in result.values)


class TestCandidatePools:
    def test_intermittent_pool_excludes_naive(self, intermittent_sku) -> None:
        pool = candidate_pool_for(intermittent_sku)
        names = [f.name for f in pool]
        assert "Croston" in names and "SBA" in names
        assert all("SeasonalNaive" not in n for n in names)

    def test_smooth_pool_includes_naive(self, smooth_sku) -> None:
        names = [f.name for f in candidate_pool_for(smooth_sku)]
        assert any("SeasonalNaive" in n for n in names)
        assert "AutoETS" in names


class TestSafeForecast:
    def test_fallback_on_degenerate_history(self) -> None:
        # SeasonalNaive on history shorter than the season -> mean fallback
        history = (2.0,) * 20
        result = safe_forecast(SeasonalNaive(period=52), history, 4)
        assert result.forecaster.endswith("mean_fallback")
        assert all(v == 2.0 for v in result.values)

    def test_fallback_on_all_zero_croston(self) -> None:
        result = safe_forecast(Croston(), (0.0,) * 15, 3)
        assert result.forecaster.endswith("mean_fallback")
        assert all(v == 0.0 for v in result.values)

    def test_no_fallback_on_valid_history(self, smooth_sku) -> None:
        result = safe_forecast(AutoETS(), smooth_sku.history, 4)
        assert result.forecaster == "AutoETS"
