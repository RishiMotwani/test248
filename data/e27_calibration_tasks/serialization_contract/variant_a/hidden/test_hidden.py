from configapp.codec import normalize


def test_unknown_keys_preserved():
    assert normalize({"known": "1", "future_key": "x"}) == {
        "known": "1", "future_key": "x"}
