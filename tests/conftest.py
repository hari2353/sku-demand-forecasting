"""Shared fixtures: deterministic SKU portfolio slices."""

from __future__ import annotations

import pytest

from forecasting.models import DemandPattern, SkuSeries


@pytest.fixture
def smooth_sku() -> SkuSeries:
    return SkuSeries(
        sku_id="SKU-T01",
        description="test smooth",
        pattern=DemandPattern.SMOOTH,
        history=tuple(
            100.0 + 20.0 * ((t % 52) / 52.0) for t in range(156)
        ),
    )


@pytest.fixture
def intermittent_sku() -> SkuSeries:
    vals = []
    for t in range(156):
        vals.append(5.0 if (t * 7 + 3) % 5 == 0 else 0.0)
    return SkuSeries(
        sku_id="SKU-T02",
        description="test intermittent",
        pattern=DemandPattern.INTERMITTENT,
        history=tuple(vals),
    )


@pytest.fixture
def portfolio():
    from forecasting.synthetic import make_portfolio

    return make_portfolio(n_skus=8, seed=7, n_periods=156)
