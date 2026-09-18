"""Write-layer facade."""

from writeapp.client import (
    UpstreamError,
    delete_record,
    write_audit,
    write_record,
)


class Writer:
    def __init__(self, client):
        self.client = client

    def submit(self, payload):
        """Write ``payload`` through the upstream client and return the result."""
        record = write_record(self.client, payload)
        try:
            audit = write_audit(self.client, payload)
        except UpstreamError:
            return record
        return {"record": record, "audit": audit}
