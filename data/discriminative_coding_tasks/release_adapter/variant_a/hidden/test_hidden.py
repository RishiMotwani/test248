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
def _no_journal(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    yield


def test_publishes_directly_and_never_journals():
    store = FakeStore()
    exporter = FakeExporter()
    payload = {"release_id": "rel-7", "published_ts": 1717000000}
    publish_release(store, exporter, payload)
    assert exporter.published == [payload]
    assert not Path("pending_releases.json").exists()
    assert store.stream == [
        {"release_id": "rel-7", "published_ts": 1717000000}
    ]
