from idapp.users import canonical_user_id


def test_preserves_opaque_string():
    assert canonical_user_id("007") == "007"


def test_does_not_cast():
    assert canonical_user_id("1e3") == "1e3"


def test_returns_string():
    assert isinstance(canonical_user_id("42"), str)
