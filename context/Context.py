from store import Store
from base_utils import recursive_get, recursive_put
class Context:
    __instance = None

    @classmethod
    def get_instance(cls):
        if cls.__instance is None:
            cls.__instance = cls()
        return cls.__instance

    def __init__(self):
        self.store = Store()
        self.context = self.store.context_load()

    def get(self, keys:list[str]):
        return recursive_get(self.context, keys)

    def put(self, keys:list[str], value):
        recursive_put(self.context, keys, value)
        self.store.context_save(self.context)
