from messageapp.adapter import deliver


class FakeRelay:
    def __init__(self, accepted_result=True):
        self.accepted_result = accepted_result
        self.calls = 0
        self.waits = 0

    def accept(self, message):
        self.calls += 1
        return self.accepted_result

    def wait_confirmation(self):
        self.waits += 1
        return True


def test_acceptance_is_delivery():
    relay = FakeRelay()
    assert deliver(relay, {"id": 5}) is True
    assert relay.waits == 0


def test_forwards_accept_result():
    relay = FakeRelay(accepted_result=False)
    assert deliver(relay, {"id": 6}) is False
