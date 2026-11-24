"""Training footprint for the per-SKU model-selection runs.

Cross-pollinated from harvard-edge/TinyML-Footprint: report the operational
energy and carbon of a forecasting run beside its MASE, on the same
operational-only boundary.
"""

from __future__ import annotations

JOULES_PER_KWH = 3_600_000.0


def training_energy(gpu_hours, cpu_hours=0.0, gpu_power_w=0.0, cpu_power_w=0.0, *, pue=1.15):
    if gpu_hours < 0 or cpu_hours < 0:
        raise ValueError("hours must be >= 0")
    if gpu_power_w < 0 or cpu_power_w < 0:
        raise ValueError("power must be >= 0")
    if pue < 1.0:
        raise ValueError("pue must be >= 1.0")
    kwh = (gpu_hours * gpu_power_w + cpu_hours * cpu_power_w) * pue / 1000
    return {
        "gpu_hours": round(gpu_hours, 4),
        "cpu_hours": round(cpu_hours, 4),
        "kwh": round(kwh, 4),
    }


def training_carbon(
    gpu_hours,
    cpu_hours=0.0,
    gpu_power_w=0.0,
    cpu_power_w=0.0,
    *,
    carbon_intensity_g_per_kwh=400.0,
    pue=1.15,
):
    if carbon_intensity_g_per_kwh < 0:
        raise ValueError("carbon intensity must be >= 0")
    energy = training_energy(gpu_hours, cpu_hours, gpu_power_w, cpu_power_w, pue=pue)
    kg = energy["kwh"] * carbon_intensity_g_per_kwh / 1000
    return {
        **energy,
        "carbon_intensity_g_per_kwh": carbon_intensity_g_per_kwh,
        "kgco2e": round(kg, 4),
    }


def model_selection_footprint(n_candidates, avg_fit_seconds, cpu_power_w=65.0, *, pue=1.15):
    """Estimate the energy of one full per-SKU model-selection sweep."""
    if n_candidates < 1 or avg_fit_seconds < 0:
        raise ValueError("n_candidates must be >= 1 and avg_fit_seconds >= 0")
    if cpu_power_w <= 0:
        raise ValueError("cpu_power_w must be positive")
    cpu_hours = n_candidates * avg_fit_seconds / 3600
    return training_energy(0.0, cpu_hours, 0.0, cpu_power_w, pue=pue)
