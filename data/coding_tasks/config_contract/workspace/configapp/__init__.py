"""configapp - config serialization/deserialization helpers.

The codec parses ``key=value`` config text, serializes mappings back to text
and merges a spec's defaults on load.
"""

from configapp.codec import ConfigCodec

__all__ = ["ConfigCodec"]