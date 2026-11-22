from forecasting.intervals import (
    conformal_half_width,
    interval_coverage,
    prediction_interval,
)


def test_half_width_is_quantile_of_magnitudes():
    residuals = [1.0, -2.0, 3.0, -4.0]
    hw = conformal_half_width(residuals, alpha=0.25)
    assert hw == 3.25  # (1-0.25)=0.75 quantile of [1,2,3,4] with interpolation


def test_prediction_interval_symmetric():
    lo, hi = prediction_interval(10.0, [1.0, 2.0, 3.0], alpha=0.1)
    assert lo < 10.0 < hi
    assert round(hi - lo, 6) == round(2 * conformal_half_width([1.0, 2.0, 3.0], alpha=0.1), 6)


def test_coverage_perfect():
    assert interval_coverage([5.0, 6.0], [4.0, 5.0], [6.0, 7.0]) == 1.0


def test_coverage_partial():
    assert interval_coverage([5.0, 10.0], [4.0, 5.0], [6.0, 7.0]) == 0.5


def test_invalid_inputs():
    try:
        conformal_half_width([])
    except ValueError:
        pass
    else:
        raise AssertionError("empty accepted")
    try:
        conformal_half_width([1.0], alpha=0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("bad alpha accepted")
    try:
        prediction_interval(1.0, [])
    except ValueError:
        pass
    else:
        raise AssertionError("empty residuals accepted")
    try:
        interval_coverage([1.0], [2.0], [3.0, 4.0])
    except ValueError:
        pass
    else:
        raise AssertionError("mismatched lengths accepted")
    try:
        interval_coverage([], [], [])
    except ValueError:
        pass
    else:
        raise AssertionError("empty accepted")
