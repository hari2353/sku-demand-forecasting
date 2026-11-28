"""Incremental forecasting (cross-pollinated from Uber incremental ETL).

Only recompute forecasts for SKUs whose data changed since the last run.
Pure Python, deterministic.
"""

from __future__ import annotations


def incremental_skus(current: dict[str, str], previous: dict[str, str]) -> list[str]:
    """Return SKUs whose fingerprint changed (or is new)."""
    changed = []
    for sku, fp in current.items():
        if previous.get(sku) != fp:
            changed.append(sku)
    return sorted(changed)


def merge_forecasts(old: dict[str, float], new: dict[str, float]) -> dict[str, float]:
    """Merge new forecasts into old, overwriting changed SKUs."""
    result = dict(old)
    result.update(new)
    return result
