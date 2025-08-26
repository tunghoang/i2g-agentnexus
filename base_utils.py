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
    
def link(url, label = 'result'):
    return f'[{label}]({PUBLISH_BASE}/{url})'

def excel_link(publish_path, label='result'):
    return link(f'{Naming.publish_path("excel-viewer", format=None)}/?file=/{publish_path}', label=label)

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
        return _allCurveRules.get(curve)
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
        return _allCurveReversedRules.get(curve, None)
    return _allCurveReversedRules

def _trim_curve_index(curve_name):
    return re.sub(":.*$", "", curve_name)

def standard_curve_name(curve):
    return getCurveReversedRules(_trim_curve_index(curve))

def aliases_of_curve(curve):
    return getCurveRules(curve)

def getLogRules(curve):
    global _allLogRules
    if _allLogRules is None:
        with open('utils/log.rules.yaml') as file:
            _allLogRules = yaml.safe_load(file)
    return _allLogRules.get(curve)

def find_similar_curves(curve, curves):
    scurve = standard_curve_name(curve)
    aliases = aliases_of_curve(scurve)
    ret_curves = [ c for c in curves if c == curve or c in aliases ]
    return ret_curves
 

