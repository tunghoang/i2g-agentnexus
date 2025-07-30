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
