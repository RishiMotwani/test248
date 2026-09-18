"""Config validation.

``AUDIT`` is a shared in-memory log used by the service layer to record
validation events. Validators may append to it.
"""

AUDIT = []


def _audit(event):
    AUDIT.append(event)


def validate_required(config, required):
    """Validate that every key in ``required`` is present in ``config``."""
    _audit(("validate_required", tuple(sorted(required))))
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"missing required keys: {missing}")
    return True
