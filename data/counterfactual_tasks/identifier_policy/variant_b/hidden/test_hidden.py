from idapp.users import canonical_user_id


def test_casts_to_int():
    assert canonical_user_id("007") == 7


def test_parses_numeric_string():
    assert canonical_user_id("42") == 42


def test_returns_int():
    assert isinstance(canonical_user_id("42"), int)
