"""Config text parsing."""


def parse_config(text):
    """Parse ``key=value`` lines into a dict, ignoring comments/blanks."""
    result = {}
    for line in str(text).splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        result[key.strip()] = value.strip()
    return result
