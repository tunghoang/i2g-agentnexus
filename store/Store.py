import os, json
import pandas as pd
from pathlib import Path
class Store:
    def __init__(self, **kwargs):
        self.store_root = kwargs.get('store_root', 'data/run_temp')
        self.store = kwargs.get('store', 'default')
    def store_path(self, filepath: str):
        return os.path.join(self.store_root, self.store, filepath)
    def prepare_dir(self, filepath):
        p = self.store_path(filepath)
        Path(p).parent.mkdir(parents=True, exist_ok=True)
    def save(self, data, filepath):
        self.prepare_dir(filepath)
        if type(data) == dict:
            with open(self.store_path(filepath)) as f:
                json.dump(data, f)
        elif isinstance(data, pd.DataFrame):
            data.to_csv(self.store_path(filepath))
        else:
            raise Exception("Dont know how to store")
    def exists(self, filepath):
        p = self.store_path(filepath)
        return os.path.isfile(p)
    def load(self, filepath):
        print(f"STORE load {filepath}")
        p = self.store_path(filepath)
        _, ext = os.path.splitext(p)
        if ext.lower() in [ '.csv' ]:
            return pd.read_csv(p, index_col=0)
        
