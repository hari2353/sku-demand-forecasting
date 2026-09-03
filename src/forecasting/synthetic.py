"""Deterministic synthetic SKU portfolio.

Covers the four demand archetypes a real demand-planning portfolio has
(the same taxonomy my Altria forecasting tool selected models for):

* smooth seasonal (weekly seasonality + gentle noise)
* trending (seasonality + strong linear growth)
* intermittent (many zeros, small demands - Croston/SBA territory)
* lumpy (many zeros + big spikes)

Seeded => every machine and every CI run produces identical portfolios.
"""

from __future__ import annotations

import math
import random

from forecasting.models import DemandPattern, SkuSeries

_PATTERNS: tuple[DemandPattern, ...] = (
    DemandPattern.SMOOTH,
    DemandPattern.TREND,
    DemandPattern.INTERMITTENT,
    DemandPattern.LUMPY,
)

_DESCRIPTIONS: dict[DemandPattern, tuple[str, ...]] = {
    DemandPattern.SMOOTH: (" staples - baseline weekly", " - core replenishment"),
    DemandPattern.TREND: (" - growth SKU (promo-driven)", " - expanding category"),
    DemandPattern.INTERMITTENT: (" - slow mover", " - spare part"),
    DemandPattern.LUMPY: (" - project-driven demand", " - bulk/b2b item"),
}


def _smooth(rng: random.Random, n: int, base: float) -> list[float]:
    out = []
    for t in range(n):
        seasonal = base * (1.0 + 0.25 * math.sin(2 * math.pi * t / 52.0))
        noise = rng.gauss(0.0, base * 0.05)
        out.append(max(0.0, seasonal + noise))
    return out


def _trend(rng: random.Random, n: int, base: float) -> list[float]:
    out = []
    for t in range(n):
        trend = base * (1.0 + 0.006 * t)
        seasonal = base * (1.0 + 0.2 * math.sin(2 * math.pi * t / 52.0))
        noise = rng.gauss(0.0, base * 0.04)
        out.append(max(0.0, trend + 0.5 * seasonal - 0.5 * base + noise))
    return out


def _intermittent(rng: random.Random, n: int) -> list[float]:
    out = []
    for _ in range(n):
        if rng.random() < 0.65:
            out.append(0.0)
        else:
            out.append(round(max(1.0, rng.gauss(4.0, 1.2)), 1))
    return out


def _lumpy(rng: random.Random, n: int) -> list[float]:
    out = []
    for _ in range(n):
        if rng.random() < 0.75:
            out.append(0.0)
        elif rng.random() < 0.85:
            out.append(round(max(1.0, rng.gauss(10.0, 3.0)), 1))
        else:
            out.append(round(max(20.0, rng.gauss(60.0, 20.0)), 1))
    return out


_GENERATORS = {
    DemandPattern.SMOOTH: lambda rng, n: _smooth(rng, n, base=rng.uniform(80.0, 140.0)),
    DemandPattern.TREND: lambda rng, n: _trend(rng, n, base=rng.uniform(60.0, 90.0)),
    DemandPattern.INTERMITTENT: lambda rng, n: _intermittent(rng, n),
    DemandPattern.LUMPY: lambda rng, n: _lumpy(rng, n),
}


def make_sku(index: int, *, seed: int = 42, n_periods: int = 156) -> SkuSeries:
    """One deterministic SKU (index -> pattern rotates across the portfolio)."""
    if n_periods < 10:
        msg = "n_periods must be >= 10"
        raise ValueError(msg)
    rng = random.Random(f"sku-{seed}-{index}")
    pattern = _PATTERNS[index % len(_PATTERNS)]
    history = _GENERATORS[pattern](rng, n_periods)
    desc_pool = _DESCRIPTIONS[pattern]
    sku_id = f"SKU-{seed:03d}{index:04d}"
    description = f"{desc_pool[index % len(desc_pool)]}".strip(" -")
    return SkuSeries(
        sku_id=sku_id,
        description=description,
        pattern=pattern,
        history=tuple(round(v, 2) for v in history),
    )


def make_portfolio(
    n_skus: int = 12, *, seed: int = 42, n_periods: int = 156
) -> list[SkuSeries]:
    """A deterministic, balanced portfolio across all four patterns."""
    if n_skus < 1:
        msg = "n_skus must be >= 1"
        raise ValueError(msg)
    return [make_sku(i, seed=seed, n_periods=n_periods) for i in range(n_skus)]


def classify(history: tuple[float, ...]) -> DemandPattern:
    """Classify a raw history via the ADI/CV^2 taxonomy (Syntetos & Boylan)."""
    if len(history) < 10:
        msg = "need >= 10 observations"
        raise ValueError(msg)
    n = len(history)
    demand_gaps = [i for i, v in enumerate(history) if v > 0]
    if len(demand_gaps) < 2:
        return DemandPattern.LUMPY
    # ADI: average inter-demand interval
    adi = n / len(demand_gaps)
    # CV^2 of non-zero demands
    nz = [v for v in history if v > 0]
    mean = sum(nz) / len(nz)
    var = sum((v - mean) ** 2 for v in nz) / (len(nz) - 1) if len(nz) > 1 else 0.0
    cv2 = var / (mean * mean) if mean > 0 else 0.0

    if adi >= 1.32 and cv2 >= 0.49:
        return DemandPattern.LUMPY
    if adi >= 1.32:
        return DemandPattern.INTERMITTENT
    if cv2 >= 0.49:
        return DemandPattern.ERRATIC
    return DemandPattern.SMOOTH
