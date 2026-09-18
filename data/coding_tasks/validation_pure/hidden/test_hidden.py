import parseapp.validator as validator
from parseapp.validator import validate_config
from parseapp.service import load_config


def test_validate_config_is_side_effect_free():
    validator.AUDIT.clear()
    assert validate_config({"a": "1", "b": "2"}) is True
    assert validator.AUDIT == [], (
        "validation must remain pure and side-effect free (no audit/db writes)")


def test_validate_config_rejects_empty_value():
    try:
        validate_config({"a": ""})
    except ValueError:
        pass
    else:
        raise AssertionError("an empty value must fail validation")


def test_load_config_runs_strict_validation():
    assert load_config("a=1\nb=2", ["a"]) == {"a": "1", "b": "2"}
    try:
        load_config("a=\nb=2", ["a"])
    except ValueError:
        pass
    else:
        raise AssertionError("load_config must reject empty values")
