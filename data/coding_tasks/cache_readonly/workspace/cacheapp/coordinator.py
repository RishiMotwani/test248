"""Cache mutation coordinator.

All cache mutation is meant to flow through ``CacheCoordinator`` so that every
write is recorded in ``write_log`` and can be audited later.
"""


class CacheCoordinator:
    def __init__(self, store):
        self.store = store
        self.write_log = []

    def set(self, key, value):
        self.store.put(key, value)
        self.write_log.append((key, value))
        return value

    def delete(self, key):
        self.store.delete(key)
        self.write_log.append((key, None))
