from cacheapp.cache_store import CacheStore
from cacheapp.coordinator import CacheCoordinator
from cacheapp.handlers import set_cached


def test_set_cached_stores_value():
    store = CacheStore()
    result = set_cached(store, "alpha", "1")
    assert result == "1"
    assert store.get("alpha") == "1"


def test_cache_mutation_routes_through_coordinator(monkeypatch):
    store = CacheStore()
    calls = []

    original_set = CacheCoordinator.set

    def spy_set(coordinator, key, value):
        calls.append((key, value))
        return original_set(coordinator, key, value)

    monkeypatch.setattr(CacheCoordinator, "set", spy_set)

    result = set_cached(store, "beta", "2")

    assert result == "2", (
        "set_cached must return the value written by CacheCoordinator.set()"
    )
    assert calls == [("beta", "2")], (
        "cache mutation must route through CacheCoordinator.set(); a direct "
        "write to store.data bypasses the coordinator and is not allowed"
    )
    assert store.get("beta") == "2"