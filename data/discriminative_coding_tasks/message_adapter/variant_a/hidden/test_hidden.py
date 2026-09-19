from messageapp.adapter import deliver


class FakeRelay:
    def __init__(self, confirmed):
        self.confirmed = confirmed
        self.accepted = []
        self.waits = 0

    def accept(self, message):
        self.accepted.append(message)
        return True

    def wait_confirmation(self):
        self.waits += 1
        return self.confirmed


def test_requires_consumer_confirmation():
    relay = FakeRelay(confirmed=False)
    try:
        deliver(relay, {"id": 1, "body": "x"})
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected failure without confirmation")
    assert relay.waits == 1


def test_returns_true_when_confirmed():
    relay = FakeRelay(confirmed=True)
    assert deliver(relay, {"id": 2}) is True
    assert relay.accepted == [{"id": 2}]
    assert relay.waits == 1
