import pytest

from writeapp.client import Client, UpstreamError
from writeapp.writer import submit


def test_no_retry_propagates_and_calls_once():
    client = Client(failures=1)
    with pytest.raises(UpstreamError):
        submit(client, {"id": 1})
    assert client.calls == 1
    assert client.records == []
