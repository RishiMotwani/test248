from writeapp.client import Client
from writeapp.writer import submit


def test_retry_once_records_one_payload():
    client = Client(failures=1)
    submit(client, {"id": 1})
    assert client.calls == 2
    assert client.records == [{"id": 1}]
