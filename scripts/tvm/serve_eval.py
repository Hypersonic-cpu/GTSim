"""Small CPU interpreter for saved online TensorIR (no simulator involvement)."""
import numpy as np
import ml_dtypes


def execute(tvm, function, inputs):
    arrays = {str(key): value for key, value in inputs.items()}
    env = {}
    types = dict(bfloat16=ml_dtypes.bfloat16, float32=np.float32, float16=np.float16,
                 int32=np.int32, int64=np.int64, bool=bool)
    binary = dict(Add=lambda a,b:a+b, Sub=lambda a,b:a-b, Mul=lambda a,b:a*b,
                  Div=lambda a,b:a/b, FloorDiv=lambda a,b:a//b, FloorMod=lambda a,b:a%b,
                  Max=max, Min=min, LT=lambda a,b:a<b, LE=lambda a,b:a<=b,
                  GT=lambda a,b:a>b, GE=lambda a,b:a>=b, EQ=lambda a,b:a==b,
                  NE=lambda a,b:a!=b, And=lambda a,b:a and b, Or=lambda a,b:a or b)

    def expr(value):
        kind = type(value).__name__
        if kind in ('IntImm', 'FloatImm'):
            return value.value
        if isinstance(value, tvm.ir.Var):
            return env[value]
        if isinstance(value, tvm.ir.TensorLoad):
            return arrays[str(value.source.name)][tuple(int(expr(index)) for index in value.indices)]
        if kind == 'Cast':
            return types[str(value.ty)](expr(value.value))
        if kind in binary:
            return binary[kind](expr(value.a), expr(value.b))
        if isinstance(value, tvm.ir.Call):
            name = value.op.name.split('.')[-1]
            if name == 'if_then_else':
                return expr(value.args[1] if expr(value.args[0]) else value.args[2])
            if name in ('exp', 'exp2', 'log', 'sqrt'):
                return getattr(np, name)(expr(value.args[0]))
        raise ValueError('Unsupported expression: ' + str(value))

    def stmt(value):
        t = tvm.tirx
        if isinstance(value, t.SeqStmt):
            for child in value.seq:
                stmt(child)
        elif isinstance(value, t.For):
            lo, size = int(expr(value.min)), int(expr(value.extent))
            for index in range(lo, lo + size):
                env[value.loop_var] = index
                stmt(value.body)
        elif isinstance(value, t.Bind):
            env[value.var] = expr(value.value)
        elif isinstance(value, t.AllocBuffer):
            buffer = value.buffer
            arrays[str(buffer.name)] = np.empty(tuple(int(n) for n in buffer.shape), dtype=types[str(buffer.dtype)])
        elif isinstance(value, t.BufferStore):
            arrays[str(value.buffer.name)][tuple(int(expr(index)) for index in value.indices)] = expr(value.value)
        elif isinstance(value, t.IfThenElse):
            if expr(value.condition):
                stmt(value.then_case)
            elif value.else_case is not None:
                stmt(value.else_case)
        elif isinstance(value, t.Evaluate):
            if type(value.value).__name__ != 'IntImm' or int(value.value) != 0:
                raise ValueError('Unknown side effect: ' + str(value))
        else:
            raise ValueError('Unsupported statement: ' + type(value).__name__)

    stmt(function.body)
    return arrays
