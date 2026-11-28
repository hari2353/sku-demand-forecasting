from forecasting.incremental import incremental_skus, merge_forecasts


def test_incremental_skus_detects_changes():
    current = {"a": "fp1", "b": "fp2", "c": "fp3"}
    previous = {"a": "fp1", "b": "fpX", "d": "fp4"}
    changed = incremental_skus(current, previous)
    assert changed == ["b", "c"]


def test_incremental_skus_none_changed():
    current = {"a": "fp1"}
    previous = {"a": "fp1"}
    assert incremental_skus(current, previous) == []


def test_merge_forecasts():
    old = {"a": 1.0, "b": 2.0}
    new = {"b": 3.0, "c": 4.0}
    merged = merge_forecasts(old, new)
    assert merged == {"a": 1.0, "b": 3.0, "c": 4.0}
    # old unchanged
    assert old == {"a": 1.0, "b": 2.0}


def test_merge_empty():
    assert merge_forecasts({}, {"a": 1.0}) == {"a": 1.0}
