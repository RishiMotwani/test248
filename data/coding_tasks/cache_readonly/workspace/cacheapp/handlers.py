"""Request handlers for the cache service."""

from cacheapp.cache_store import CacheStore


def handle_get(store: CacheStore, key):
    return store.get(key)


def set_cached(store: CacheStore, key, value):
    """Store ``value`` under ``key`` and return the stored value."""
    raise NotImplementedError("set_cached is not implemented yet")
