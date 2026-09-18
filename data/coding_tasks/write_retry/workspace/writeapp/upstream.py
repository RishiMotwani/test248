"""Upstream record sink.

The upstream write is **not idempotent**: sending the same payload twice creates
two records. Any retry of a failed write therefore risks a duplicate.
"""


class UpstreamError(Exception):
    """Raised when the upstream write fails (including transient failures)."""


class UpstreamClient:
    def __init__(self):
        self.calls = 0
        self.should_fail = False

    def write(self, payload):
        self.calls += 1
        if self.should_fail:
            raise UpstreamError("upstream write failed")
        return {"ok": True, "payload": payload}
