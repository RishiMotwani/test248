"""In-memory cache store.

The store owns the raw ``data`` dict. Mutations go through ``put`` so that the
coordinator can audit them; request handlers are expected to treat the store as
read-only.
"""


class CacheStore:
    def __init__(self):
        self.data = {}
        self.put_count = 0

    def put(self, key, value):
        self.data[key] = value
        self.put_count += 1

    def get(self, key, default=None):
        return self.data.get(key, default)

    def delete(self, key):
        self.data.pop(key, None)
