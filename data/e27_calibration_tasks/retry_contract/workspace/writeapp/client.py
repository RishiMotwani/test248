"""Low-level write client shared by the submission contract."""


class UpstreamError(Exception):
    """Raised when the upstream write endpoint rejects a payload."""
    pass


class Client:
    """Minimal write client; the first ``failures`` writes raise."""

    def __init__(self, failures=0):
        self.failures = failures
        self.calls = 0
        self.records = []

    def write(self, payload):
        self.calls += 1
        if self.calls <= self.failures:
            raise UpstreamError("upstream unavailable")
        self.records.append(payload)
