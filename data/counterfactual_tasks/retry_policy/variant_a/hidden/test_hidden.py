from writeapp.client import Client, UpstreamError
from writeapp.writer import submit


def test_single_attempt_no_retry():
    client = Client(failures=1)
    try:
        submit(client, {"id": 1})
    except UpstreamError:
        pass
    else:
        raise AssertionError("expected UpstreamError")
    assert client.calls == 1
    assert client.records == []


def test_success_records_payload():
    client = Client(failures=0)
    result = submit(client, {"id": 2})
    assert result == {"status": "ok", "payload": {"id": 2}}
    assert client.calls == 1
    assert client.records == [{"id": 2}]
