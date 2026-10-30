"""Explicit weight-only HBF placement on semantic and scheduled TensorIR."""
import json

KEY = 'gtsim.semantic_tensor_tiers'


def is_weight(name):
    return name.startswith('W_') or name in ('A_log', 'dt_bias')


def annotate(tvm, module, topology):
    if topology not in ('FullHBM', 'CoLocated'):
        raise ValueError('serving experiment requires FullHBM or CoLocated')
    result = tvm.IRModule(dict(module.functions))
    for name, function in module.functions.items():
        tiers = {str(p.name): ('HBF' if topology == 'CoLocated' and is_weight(str(p.name)) else 'HBM')
                 for p in function.params}
        result[name] = function.with_attr(KEY, json.dumps(tiers, sort_keys=True))
    return result


def validate(function, hardware):
    if not function.attrs or KEY not in function.attrs:
        raise ValueError('missing explicit TensorIR tensor-tier manifest')
    tiers = json.loads(str(function.attrs[KEY]))
    names = {str(p.name) for p in function.params}
    if set(tiers) != names:
        raise ValueError('TensorIR placement manifest does not cover exactly all parameters')
    for name, tier in tiers.items():
        expected = 'HBF' if hardware['topology'] == 'CoLocated' and is_weight(name) else 'HBM'
        if tier != expected:
            raise ValueError('weight-only HBF placement mismatch or unavailable device: ' + name)
    return tiers
