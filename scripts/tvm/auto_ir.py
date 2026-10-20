"""Small structural IR utilities for the pinned TVM compiler extensions."""

import tvm_ffi


def substitute(tvm, obj, values):
    return tvm_ffi.structural_map(obj, (tvm.ir.Var, lambda var: values.get(var, var)), order="post")


def nodes(obj, kind):
    result = []
    tvm_ffi.structural_walk(obj, (kind, lambda node: result.append(node)))
    return result


def mutate(obj, kind, callback):
    return tvm_ffi.structural_mutate(obj, (kind, callback))


def scalar(tvm, expr, env=None):
    if env:
        expr = substitute(tvm, expr, {var: tvm.tirx.const(value, 'int32') for var, value in env.items()})
    expr = tvm.sym.Analyzer().simplify(expr)
    if isinstance(expr, tvm.tirx.IntImm):
        return int(expr)
    if type(expr).__name__ in ('BitwiseAnd','BitwiseOr'):
        a,b = scalar(tvm,expr.a),scalar(tvm,expr.b)
        return a & b if type(expr).__name__ == 'BitwiseAnd' else a | b
    raise ValueError(f"nonconstant bounded expression: {expr}")


def address(tvm, call):
    while isinstance(call, tvm.ir.Call):
        call = call.args[0]
    if not isinstance(call, tvm.ir.TensorLoad):
        raise ValueError(f"expected buffer address, found {call}")
    return call.source, list(call.indices)


def seq(tvm, items):
    items = [item for item in items if item is not None]
    return items[0] if len(items)==1 else tvm.tirx.SeqStmt(items) if items else tvm.tirx.Evaluate(0)


def trace_json(trace):
    """Portable instruction/decision JSON, not IR serialization of callbacks."""
    from tvm_ffi.container import Array,Map
    def convert(value):
        if value is None or isinstance(value,(str,int,float,bool)):
            return value
        if isinstance(value,(list,tuple,Array)):
            return [convert(item) for item in value]
        if isinstance(value,Map):
            return {convert(key):convert(item) for key,item in value.items()}
        if hasattr(value,'value'):
            return value.value
        raise TypeError(f'unsupported trace JSON value {type(value).__name__}')
    return convert(trace.as_json())
