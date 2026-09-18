from writeapp.client import (
    UpstreamError,
    create_client,
    delete_record,
    write_audit,
    write_record,
)
from writeapp.writer import Writer


def test_success_commits_both_writes():
    client = create_client()
    result = Writer(client).submit({"id": 1})
    assert result["record"]["status"] == "recorded"
    assert result["audit"]["status"] == "audited"
    assert len(client.records) == 1
    assert len(client.audit_log) == 1


def test_failure_after_first_write_leaves_no_partial_state():
    client = create_client(fail_on=["audit"])
    try:
        Writer(client).submit({"id": 2})
    except UpstreamError:
        pass
    else:
        raise AssertionError(
            "submit must surface the upstream failure to the caller")
    assert client.records == [], (
        "no partial state is allowed: the first write must be rolled back when "
        "the second write fails")
    assert client.audit_log == [], "the failed audit recorded nothing"


def test_single_attempt_no_automatic_retry():
    client = create_client(fail_on=["record"], fail_times=1)
    try:
        Writer(client).submit({"id": 3})
    except UpstreamError:
        pass
    else:
        raise AssertionError(
            "submit must surface the upstream failure to the caller")
    assert client.records == []
    assert client.total_calls == 1, (
        "a failed submit must not be retried automatically: a retry would "
        "duplicate committed rows")