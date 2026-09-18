"""Service layer that loads and validates configuration."""

from parseapp.parser import parse_config
from parseapp.validator import validate_required


def load_config(text, required):
    """Parse ``text``, validate it, and return the parsed config."""
    config = parse_config(text)
    validate_required(config, required)
    return config
