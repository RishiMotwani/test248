"""Fake upstream client for the write layer.

``write_record`` and ``write_audit`` model an upstream that records two side
effects per submit. Failures surface as ``UpstreamError``.
"""


class UpstreamError(RuntimeError):
    pass


class FakeUpstream:
    def __init__(self, fail_on=(), fail_times=1):
        self.fail_on = list(fail_on)
        self.fail_times = fail_times
        self._failures_left = {op: fail_times for op in self.fail_on}
        self.records = []
        self.audit_log = []
        self.total_calls = 0

    def _tick(self, op):
        self.total_calls += 1
        if self._failures_left.get(op, 0) > 0:
            self._failures_left[op] -= 1
            raise UpstreamError(f"{op} failed")


def create_client(fail_on=(), fail_times=1):
    return FakeUpstream(fail_on=list(fail_on), fail_times=fail_times)


def write_record(client, payload):
    client._tick("record")
    client.records.append(payload)
    return {"status": "recorded", "payload_id": payload.get("id")}


def write_audit(client, payload):
    client._tick("audit")
    client.audit_log.append(payload.get("id"))
    return {"status": "audited", "payload_id": payload.get("id")}


def delete_record(client, payload):
    client._tick("delete_record")
    if payload in client.records:
        client.records.remove(payload)