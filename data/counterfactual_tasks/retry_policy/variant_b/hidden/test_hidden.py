from writeapp.client import Client, UpstreamError
from writeapp.writer import submit


def test_retries_exactly_once():
    client = Client(failures=1)
    result = submit(client, {"id": 1})
    assert result == {"status": "ok", "payload": {"id": 1}}
    assert client.calls == 2
    assert client.records == [{"id": 1}]


def test_surfaces_error_after_two_failures():
    client = Client(failures=2)
    try:
        submit(client, {"id": 3})
    except UpstreamError:
        pass
    else:
        raise AssertionError("expected UpstreamError")
    assert client.calls == 2
    assert client.records == []
