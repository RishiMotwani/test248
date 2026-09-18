from configapp.codec import ConfigCodec

CODEC = ConfigCodec({"host": "localhost", "port": "8080", "timeout": "30"})


def test_unknown_field_round_trips():
    text = "host=alpha\nfuture_feature=enabled\n"
    cfg = CODEC.load(text)
    assert cfg["future_feature"] == "enabled", (
        "unknown fields are deliberately preserved so configs written by "
        "newer versions survive older-relaying versions"
    )
    out = CODEC.serialize(cfg)
    assert "future_feature=enabled" in out


def test_explicit_empty_string_survives():
    cfg = CODEC.load("host=\nport=8080\n")
    assert cfg["host"] == "", (
        "an explicitly empty string is a valid value and must not be replaced "
        "by the default")
    assert "host=" in CODEC.serialize(cfg), (
        "an explicit empty string must survive serialization as host=")


def test_absent_field_gets_default():
    cfg = CODEC.load("host=alpha\n")
    assert cfg["port"] == "8080"
    assert cfg["timeout"] == "30"
    out = CODEC.serialize(cfg)
    assert "port=8080" in out
    assert "timeout=30" in out