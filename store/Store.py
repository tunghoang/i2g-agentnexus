import os, json
import traceback
import pandas as pd
from naming import Naming
from pathlib import Path
class Store:
    def __init__(self, **kwargs):
        self.store_root = kwargs.get('store_root', 'data/run_temp')
        self.store = kwargs.get('store', 'default')

    def context_path(self):
        #return 'context.json'
        return Naming.context_path()
    def store_path(self, filepath: str):
        return os.path.join(self.store_root, self.store, filepath)
    def prepare_dir(self, filepath):
        p = self.store_path(filepath)
        Path(p).parent.mkdir(parents=True, exist_ok=True)

    def save(self, data, filepath):
        self.prepare_dir(filepath)
        if type(data) == dict:
            with open(self.store_path(filepath), 'w') as f:
                json.dump(data, f)
        elif isinstance(data, pd.DataFrame):
            data.to_csv(self.store_path(filepath))
        else:
            raise Exception("Dont know how to store")

    def context_save(self, data):
        filepath = self.context_path()
        self.save(data, filepath)

    def exists(self, filepath):
        p = self.store_path(filepath)
        return os.path.isfile(p)
    def load(self, filepath):
        print(f"STORE load {filepath}")
        p = self.store_path(filepath)
        _, ext = os.path.splitext(p)
        if ext.lower() in [ '.csv' ]:
            return pd.read_csv(p, index_col=0)
        elif ext.lower() in [ '.json' ]:
            with open(p) as f:
                data = json.load(f)
            return data
        
    def context_load(self):
        filepath = self.context_path()
        if not self.exists(filepath):
            self.context_save({})
        return self.load(filepath)

    def get_curves_in_well(self, well):
        try:
            curves_df = self.load(f'wells/{well}.csv')
            return list(curves_df['curve'].unique())
        except Exception as e:
            #traceback.print_exc()
            print(str(e))
            return None

    def save_curves_in_well(self, curves: list[dict], well: str):
        curves_df = pd.DataFrame()
        try:
            curves_df = self.load(f'wells/{well}.csv')
        except:
            pass
        new_curves_df = pd.DataFrame(curves)
        curves_df = pd.concat([curves_df, new_curves_df], ignore_index = True)
        curves_df = curves_df.drop_duplicates()
        self.save(curves_df, f'wells/{well}.csv')

