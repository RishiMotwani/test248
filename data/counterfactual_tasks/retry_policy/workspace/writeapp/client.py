id="qy70hb"
class UpstreamError(RuntimeError):
    pass


class Client:
    def __init__(self, failures=0):
        self.failures = int(failures)
        self.calls = 0
        self.records = []

    def write(self, payload):
        self.calls += 1

        if self.failures > 0:
            self.failures -= 1
            raise UpstreamError("transient failure")

        self.records.append(payload)
        return {"status": "ok", "payload": payload}
