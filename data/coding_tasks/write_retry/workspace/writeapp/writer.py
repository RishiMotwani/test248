"""Writer layer on top of the upstream sink."""

from writeapp.upstream import UpstreamError


class WriteFailed(Exception):
    """Raised when a record could not be written."""


def write_record(client, payload):
    """Send a single record to ``client`` and return the upstream result."""
    return client.write(payload)


def submit(client, payload):
    """Submit ``payload`` for writing, surfacing failures to the caller.

    Must return the upstream result on success. On ``UpstreamError`` it must
    raise ``WriteFailed``.
    """
    raise NotImplementedError("submit is not implemented yet")
