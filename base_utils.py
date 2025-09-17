import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime, timedelta, timezone

from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
from mlflow.artifacts import download_artifacts

import json
import yaml
import re
from naming import Naming
#PUBLISH_BASE="http://dashboard.portal:9999"
PUBLISH_BASE="http://dashboard.portal:8990"
def __do_get(d, key):
    if type(key) == int and type(d) == list:
        return d[key]
    if type(d) != dict:
        return None
    return d.get(f'{key}')

def recursive_get(tree:dict, path: list):
    if type(tree) not in [dict, list]:
        return None
    subtree = __do_get(tree, path[0])
    if len(path) == 1:
        return subtree
    return recursive_get(subtree, path[1:])

def recursive_put(tree: dict, path: list, value):
    _tree = tree
    for p in path[:-1]:
        if p not in _tree:
            _tree[p] = {}
        _tree = _tree[p]
    _tree[ path[-1] ] = value

def update_dict(d, d1):
    for k in d1:
        d[k] = d1[k]

def iframe(url, height='960px'):
    if height is None:
        return f'<iframe width="100%" src="{PUBLISH_BASE}/{url}"></iframe>'
    return f'<iframe width="100%" height="{height}" src="{PUBLISH_BASE}/{url}"></iframe>'
    
def link(url, label = 'result', syntax='md'):
    if syntax == 'md':
        return f'[{label}]({PUBLISH_BASE}/{url})'
    if syntax == 'html':
        return f'<a href="{PUBLISH_BASE}/{url}" target="_blank">{label}</a>'

def excel_link(publish_path, label='result', syntax='md'):
    return link(f'{Naming.publish_path("excel-viewer", format=None)}/?file=/{publish_path}', label=label, syntax=syntax)

def normalize(s):
    minV = s.min()
    maxV = s.max()
    s1 = (s - minV) / (maxV - minV)
    if maxV - minV == 0:
        s1[:] = 1.0
    return s1

_allCurveRules = None
_allCurveReversedRules = None
_allLogRules = None
_allFlagRules = None
_allCurveUnits = None
def getUnit(curve = None):
    global _allCurveUnits
    if _allCurveUnits is None:
        with open('utils/curve.units.yaml') as file:
            _allCurveUnits = yaml.safe_load(file)
    if curve:
        return _allCurveUnits.get(curve)
    return _allCurveUnits

def getCurveRules(curve = None):
    global _allCurveRules
    if _allCurveRules is None:
        with open('utils/curve.rules.yaml') as file:
            _allCurveRules = yaml.safe_load(file)
    if curve:
        return _allCurveRules.get(curve, [])
    return _allCurveRules

def getCurveReversedRules(curve = None):
    global _allCurveReversedRules
    if _allCurveReversedRules is None:
        allCurveRules = getCurveRules()
        _allCurveReversedRules = {}
        for c, aliases in allCurveRules.items():
            for alias in aliases:
                _allCurveReversedRules[alias] = c
    if curve:
        return _allCurveReversedRules.get(curve, curve)
    return _allCurveReversedRules

def _trim_curve_index(curve_name):
    return re.sub(":.*$", "", curve_name)

def standard_curve_name(curve):
    sName = getCurveReversedRules(_trim_curve_index(curve))
    return sName

def aliases_of_curve(curve):
    return getCurveRules(curve)

def getLogRules(curve):
    global _allLogRules
    if _allLogRules is None:
        with open('utils/log.rules.yaml') as file:
            _allLogRules = yaml.safe_load(file)
    return _allLogRules.get(curve)

def getFlagRules(curve):
    global _allFlagRules
    if _allFlagRules is None:
        with open('utils/flag.rules.yaml') as file:
            _allFlagRules = yaml.safe_load(file)
    return _allFlagRules.get(curve)

def find_similar_curves(curve, curves):
    scurve = standard_curve_name(curve)
    aliases = aliases_of_curve(scurve)
    ret_curves = [ c for c in curves if c == curve or c in aliases ]
    return ret_curves
 

# Hàm tính MAPE
def calculate_mape(y_true, y_pred):
    # Tránh chia cho 0 bằng cách thêm epsilon nhỏ
    mask = y_true != 0  # Chỉ tính MAPE cho các giá trị khác 0
    if np.sum(mask) == 0:
        return np.nan  # Trả về NaN nếu tất cả y_true là 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

# Hàm tính các chỉ số đánh giá
def compute_metrics(y_true, y_pred):
    print("SHAPES:", y_true.shape, y_pred.shape)
    return {
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAPE': calculate_mape(y_true, y_pred),
        'R2': r2_score(y_true, y_pred)
    }

def get_mlflow_experiment(client: MlflowClient, name: str):
    experiment = client.get_experiment_by_name(name)
    if not experiment:
        experiment_id = client.create_experiment(name)
        experiment = client.get_experiment(experiment_id)
    return experiment

def human_readable_diff(start_time: datetime, end_time: datetime) -> str:
    if not start_time or not end_time:
        return "N/A"

    delta = end_time - start_time
    if delta.days > 3:
        utc7 = timezone(timedelta(hours=7))
        return start_time.astimezone(utc7).strftime("%d/%m/%Y, %I:%M:%S %p")

    seconds_total = int(delta.total_seconds())
    minutes, seconds = divmod(seconds_total, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)

    if days:
        return f"{days} days ago"
    if hours:
        return f"{hours} hours ago"
    if minutes:
        return f"{minutes} minutes ago"
    
    return f"{seconds} seconds ago"

def parse_json_param(param: str):
    if not param:
        return "N/A"
    try:
        result = json.loads(param)
        if type(result) == list:
            return ", ".join(result)
        elif type(result) == dict:
            return ", ".join(f"{k}: {v}" for k, v in result.items())
        return result
    except:
        return param

def parse_float_param(param: float | int):
    if param is None:
        return "N/A"
    return f"{param:.2f}"

def get_mlflow_artifact_path(
        experiment_id: str,
        run_id: str,
        artifact_path: str,
        allow_remote: bool = False,
    ) -> str | None:

    mlflow_path = Naming.mlflow_path(f"{experiment_id}/{run_id}/artifacts")
    local_file_path = os.path.join(mlflow_path, artifact_path)
    if os.path.isfile(local_file_path):
        return local_file_path 

    # MLflow is running on a different host
    if allow_remote:
        try:
            os.makedirs(dest_path, exist_ok=True)
            downloaded_path = download_artifacts(
                run_id=run_id,
                artifact_path=artifact_path,
                dst_path=dest_path
            )
            return downloaded_path
        except Exception:
            return None

    return None

def seconds_ago_to_timestamp(seconds_ago: int) -> int:
    dt = datetime.now() - timedelta(seconds=seconds_ago)
    return int(dt.timestamp()) * 1000
