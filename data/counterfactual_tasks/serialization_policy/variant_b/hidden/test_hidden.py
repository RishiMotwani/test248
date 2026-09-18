from configapp.codec import normalize


def test_drops_unknown_keys():
    result = normalize({"known": "1", "future_key": "x"})
    assert result == {"known": "1"}


def test_unknown_only_is_empty():
    assert normalize({"future_key": "x"}) == {}
