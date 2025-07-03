class MemoryCache:
    instance = None
    def __init__(self):
        self._cache = dict()

    @classmethod
    def get_instance(cls):
        if cls.instance is None:
            cls.instance = MemoryCache()
        return cls.instance

    def get(self, key):
        return self._cache.get(key, None)

    def put(self, key, value):
        self._cache[key] = value
