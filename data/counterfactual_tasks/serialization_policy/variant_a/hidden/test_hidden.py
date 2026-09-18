from configapp.codec import normalize


def test_preserves_unknown_keys():
    result = normalize({"known": "1", "future_key": "x"})
    assert result == {"known": "1", "future_key": "x"}


def test_known_keys_kept():
    assert normalize({"known": "1"}) == {"known": "1"}
