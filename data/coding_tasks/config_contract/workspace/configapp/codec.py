"""Config parsing, serialization and default merging."""


def parse(text):
    """Parse ``key=value`` lines into a plain dict (comments/blanks ignored)."""
    result = {}
    for line in str(text).splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        result[key.strip()] = value.strip()
    return result


def serialize(mapping):
    """Serialize a mapping into ``key=value`` lines, sorted by key."""
    lines = []
    for key in sorted(mapping):
        lines.append(f"{key}={mapping[key]}")
    return "\n".join(lines)


class ConfigCodec:
    """Round-trips config text against an optional field spec."""

    def __init__(self, spec=None):
        self.spec = spec or {}

    def parse(self, text):
        return parse(text)

    def serialize(self, mapping):
        return serialize(mapping)

    def load(self, text, spec=None):
        """Parse ``text`` and merge the spec's defaults into the result."""
        spec = dict(spec or self.spec or {})
        parsed = self.parse(text)
        merged = {}
        for key, default in spec.items():
            if key in parsed and parsed[key] == "":
                value = default
            else:
                value = parsed.get(key, default)
            merged[key] = value
        for key in parsed:
            if key not in merged:
                merged[key] = parsed[key]
        return merged
