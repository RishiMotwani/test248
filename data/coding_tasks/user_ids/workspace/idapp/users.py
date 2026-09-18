"""User identity helpers.

``normalize_user_id`` is the single place where a client-supplied user id is
turned into the canonical value used as the storage key.
"""


def normalize_user_id(value):
    """Return the canonical representation of a client-supplied user id."""
    return int(value)
