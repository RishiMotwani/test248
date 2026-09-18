from cacheapp.cache_store import CacheStore
from cacheapp.handlers import set_cached


def test_set_cached_stores_value():
    store = CacheStore()
    result = set_cached(store, "alpha", "1")
    assert result == "1"
    assert store.get("alpha") == "1"


def test_cache_mutation_routes_through_put():
    store = CacheStore()
    set_cached(store, "beta", "2")
    assert store.put_count == 1, (
        "cache mutation must go through the coordinator/put() path, never a "
        "direct write to store.data")
