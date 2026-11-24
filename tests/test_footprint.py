from forecasting.footprint import model_selection_footprint, training_carbon, training_energy


def test_energy_math():
    out = training_energy(2.0, 1.0, 400.0, 65.0, pue=1.0)
    assert out["kwh"] == round((2.0 * 400 + 1.0 * 65) / 1000, 4)


def test_carbon_math():
    out = training_carbon(2.0, 0.0, 400.0, 0.0, pue=1.0, carbon_intensity_g_per_kwh=400)
    assert out["kgco2e"] == 0.32


def test_model_selection_footprint():
    out = model_selection_footprint(10, 30.0, cpu_power_w=65.0, pue=1.0)
    assert out["kwh"] == round(10 * 30 * 65 / 3600 / 1000, 4)


def test_validation():
    for kwargs in (
        {"gpu_hours": -1, "cpu_hours": 0, "gpu_power_w": 10, "cpu_power_w": 0},
        {"gpu_hours": 1, "cpu_hours": 0, "gpu_power_w": 10, "cpu_power_w": 0, "pue": 0.9},
        {
            "gpu_hours": 1,
            "cpu_hours": 0,
            "gpu_power_w": 10,
            "cpu_power_w": 0,
            "carbon_intensity_g_per_kwh": -1,
        },
    ):
        try:
            training_carbon(**kwargs)
        except ValueError:
            continue
        raise AssertionError(f"accepted invalid input: {kwargs}")
    for args in ((0, 1.0), (-1, 1.0), (1, -1.0)):
        try:
            model_selection_footprint(*args)
        except ValueError:
            continue
        raise AssertionError(f"accepted invalid input: {args}")
