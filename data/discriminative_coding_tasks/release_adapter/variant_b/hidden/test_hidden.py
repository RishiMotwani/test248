import json
from pathlib import Path

import pytest

from releaseapp.adapter import publish_release


class FakeExporter:
    def __init__(self):
        self.published = []

    def export(self, release):
        self.published.append(release)


class FakeStore:
    def __init__(self):
        self.stream = []

    def write_audit(self, entry):
        self.stream.append(entry)


@pytest.fixture(autouse=True)
def _sandbox(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    yield


def test_journals_raw_payload_before_export():
    store = FakeStore()
    exporter = FakeExporter()
    payload = {"release_id": "rel-9", "published_ts": 1717100000}
    publish_release(store, exporter, payload)
    assert exporter.published == [payload]
    lines = Path("pending_releases.json").read_text().splitlines()
    assert json.loads(lines[0]) == payload
    assert store.stream == [
        {"release_id": "rel-9", "published_ts": 1717100000, "journal": "pending_releases.json"}
    ]
