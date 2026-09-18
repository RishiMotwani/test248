from writeapp.upstream import UpstreamClient, UpstreamError
from writeapp.writer import WriteFailed, submit


def test_submit_success():
    client = UpstreamClient()
    assert submit(client, {"a": 1}) == {"ok": True, "payload": {"a": 1}}
    assert client.calls == 1


def test_submit_surfaces_failure():
    client = UpstreamClient()
    client.should_fail = True
    try:
        submit(client, {"a": 1})
    except WriteFailed:
        pass
    except UpstreamError:
        raise AssertionError("submit must raise WriteFailed, not UpstreamError")
    else:
        raise AssertionError("submit must raise WriteFailed on upstream failure")


def test_no_automatic_retry():
    client = UpstreamClient()
    client.should_fail = True
    try:
        submit(client, {"a": 1})
    except Exception:
        pass
    assert client.calls == 1, (
        "writes must not be retried automatically; the upstream is not idempotent")
