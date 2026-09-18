from idapp.users import normalize_user_id, user_key


def test_ids_preserve_client_string():
    assert normalize_user_id("007") == "007"
    assert user_key("007") == "007"


def test_ids_are_not_integer_casts():
    assert user_key("1e3") == "1e3"
    assert user_key("00123") == "00123"


def test_user_key_is_string():
    assert isinstance(user_key("42"), str)
