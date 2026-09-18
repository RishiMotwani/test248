"""writeapp - write-layer facade over a fake upstream client.

Upstream mutations go through `write_record` and `write_audit`; failures
surface as `UpstreamError`.
"""

from writeapp.writer import Writer

__all__ = ["Writer"]