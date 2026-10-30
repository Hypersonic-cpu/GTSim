"""Scheduled mathematical TensorIR -> explicit Ampere/SIMD tile graph.

This is a compiler extension for the simulator, not a CUDA performance claim.
It consumes serialized IR, derives regions from its expressions and bounds,
and retains full-completion kernel boundaries and register-ring reuse edges.
"""
from collections import Counter
import math

import numpy as np

from auto_ir import nodes
from serve_rules import stages


def dtype_name(dtype):
    return {'bfloat16': 'bf16', 'float16': 'fp16', 'float32': 'fp32'}[str(dtype)]


class Graph:
    def __init__(self, hardware):
        self.hardware = hardware
        self.tensors = []
        self.ctas = []
        self.operations = []
        self.tensor_ids = {}

    def tensor(self, name, shape, dtype='float32', tier='HBM'):
        shape = tuple(map(int, shape))
        shape = (math.prod(shape[:-1]), shape[-1]) if len(shape) > 1 else (1, shape[0])
        if name in self.tensor_ids:
            ident = self.tensor_ids[name]
            if self.tensors[ident][2:] != (shape, dtype_name(dtype), tier):
                raise ValueError('inconsistent reused tensor identity: ' + name)
            return ident
        ident = len(self.tensors)
        self.tensor_ids[name] = ident
        self.tensors.append((ident, name, shape, dtype_name(dtype), tier))
        return ident

    def cta(self, warps, shared, registers):
        if shared > min(self.hardware['shared_memory_per_cta_bytes'], self.hardware['shared_memory_per_sm_bytes']):
            raise ValueError('shared-memory capacity exceeded')
        if registers > min(self.hardware['register_file_per_sm_bytes'], warps * 32 * 255 * 4):
            raise ValueError('register capacity exceeded')
        ident = len(self.ctas)
        self.ctas.append((ident, ident % self.hardware['num_sms'], warps, shared, max(registers, 128 * warps)))
        return ident

    def add(self, cta, warp, op, tile, deps=(), tensor=0, dtype='float32', stage='kernel', granularity=16, layout='row'):
        ident = len(self.operations)
        self.operations.append((op, cta, warp, tensor, dtype_name(dtype), tuple(tile),
                                tuple(dict.fromkeys(deps)), stage, granularity, layout))
        return ident

    def write(self, path):
        # Stream to avoid constructing a second full textual graph in memory.
        def vec(values):
            return '[' + ','.join(map(str, values)) + ']'
        with open(path, 'w') as handle:
            handle.write('gtsim.tile_ir 2\n')
            for ident, name, shape, dtype, tier in self.tensors:
                handle.write(f'tensor id={ident} name={name} shape={vec(shape)} strides=[{shape[1]},1] dtype={dtype} tier={tier}\n')
            for ident, sm, warps, shared, registers in self.ctas:
                handle.write(f'cta id={ident} sm={sm} warps={warps} shared={shared} registers={registers}\n')
            for ident, (op, cta, warp, tensor, dtype, tile, deps, stage, granularity, layout) in enumerate(self.operations):
                handle.write(f'%{ident} = {op} name={stage}__{ident} cta={cta} warp={warp} tensor={tensor} dtype={dtype} '
                             f'tile={vec(tile)} after_issue=[] after_complete={vec(deps)} granularity={granularity} layout={layout}\n')


def evaluate(tvm, mod, inputs):
    """Execute SBlock mathematics independently of loop tiling on small inputs."""
    import ml_dtypes
    types = dict(bfloat16=ml_dtypes.bfloat16, float16=np.float16, float32=np.float32)
    arrays = {key: np.asarray(value) for key, value in inputs.items()}
    functions = dict(Add=np.add, Sub=np.subtract, Mul=np.multiply, Div=np.divide,
                     FloorDiv=np.floor_divide, FloorMod=np.remainder,
                     Max=np.maximum, Min=np.minimum, LT=np.less, LE=np.less_equal,
                     GT=np.greater, GE=np.greater_equal, EQ=np.equal, NE=np.not_equal)
    for block in nodes(mod, tvm.s_tir.SBlock):
        if block.name_hint == 'root':
            continue
        output = block.writes[0].source
        shape = tuple(int(n) for n in output.shape)
        variables = list(block.iter_vars)
        env = {}
        for i, var in enumerate(variables):
            n = int(var.dom.extent)
            dims = [1] * len(variables)
            dims[i] = n
            env[var.var] = np.arange(n).reshape(dims)
        reduction = tuple(i for i, var in enumerate(variables) if int(var.iter_type) == 2)
        store = nodes(block.body, tvm.tirx.BufferStore)[0]

        def expr(value):
            kind = type(value).__name__
            if kind in ('IntImm', 'FloatImm'):
                return value.value
            if isinstance(value, tvm.ir.Var):
                return env[value]
            if kind == 'Cast':
                return np.asarray(expr(value.value)).astype(types[str(value.ty)])
            if isinstance(value, tvm.ir.TensorLoad):
                if value.source.same_as(output):
                    return 0.0
                source = arrays[str(value.source.name)]
                indices = tuple(expr(index) for index in value.indices)
                # if_then_else is lazy in TIR; vector indexing clips only the
                # inactive arm, whose result is subsequently discarded.
                indices = tuple(np.clip(index, 0, source.shape[i] - 1) for i, index in enumerate(indices))
                return source[indices].astype(np.float32)
            if kind in functions:
                return functions[kind](expr(value.a), expr(value.b))
            if isinstance(value, tvm.ir.Call):
                name = value.op.name.split('.')[-1]
                if name == 'if_then_else':
                    return np.where(expr(value.args[0]), expr(value.args[1]), expr(value.args[2]))
                unary = dict(exp=np.exp, exp2=np.exp2, log=np.log, sqrt=np.sqrt,
                             rsqrt=lambda x: 1 / np.sqrt(x), abs=np.abs, fabs=np.abs)
                if name in unary:
                    return unary[name](expr(value.args[0]))
            raise ValueError('unsupported numerical expression: ' + kind + ' ' + str(value))

        value = expr(store.value)
        if reduction:
            extents = tuple(int(var.dom.extent) for var in variables)
            value = np.sum(np.broadcast_to(value, extents), axis=reduction)
        arrays[str(output.name)] = np.asarray(value).reshape(shape).astype(types[str(output.dtype)])
    return arrays


def lower_vector(tvm, mod, hardware):
    from serve_memory import validate
    declared_tiers = validate(mod['main'], hardware)
    graph = Graph(hardware)
    scheduled = stages(tvm, mod)
    buffers = {}
    for spec in scheduled:
        block = spec['block']
        for region in [*block.reads, *block.writes]:
            buffers[str(region.source.name)] = region.source
    for name, buffer in buffers.items():
        tier = declared_tiers.get(name, 'HBM')
        graph.tensor(name, buffer.shape, buffer.dtype, tier)
    completed = []
    report = []
    operation_map = dict(Add='simd.add', Sub='simd.sub', Mul='simd.mul', Div='sfu.divide',
                         Max='simd.max', Min='simd.min', Cast='simd.cast')
    for spec in scheduled:
        block, tile_n, warps, tile_k, depth = (spec[key] for key in ('block', 'output_tile', 'warps', 'reduction_tile', 'stages'))
        output = block.writes[0].source
        shape = tuple(int(n) for n in output.shape)
        total, columns = math.prod(shape), shape[-1]
        variables = list(block.iter_vars)
        spatial = [v for v in variables if int(v.iter_type) == 0]
        red = [v for v in variables if int(v.iter_type) == 2]
        reduction = int(red[0].dom.extent) if red else 1
        store = nodes(block.body, tvm.tirx.BufferStore)[0]
        expression = store.value
        fma = False
        if red:
            if type(expression).__name__ != 'Add' or not isinstance(expression.a, tvm.ir.TensorLoad) or not expression.a.source.same_as(output):
                raise ValueError('only additive vector reductions are supported')
            expression = expression.b
            fma = type(expression).__name__ == 'Mul'
        loads = []
        for load in nodes(expression, tvm.ir.TensorLoad):
            if all(not tvm_compatible_equal(tvm, load, previous) for previous in loads):
                loads.append(load)
        counts = Counter()
        seen = set()
        def count_expr(value, skip_multiply=False):
            token = tvm.ir.save_json(value)
            if token in seen:
                return
            seen.add(token)
            kind = type(value).__name__
            if isinstance(value, tvm.ir.TensorLoad):
                return
            if kind in operation_map:
                if not (skip_multiply and kind == 'Mul'):
                    counts[operation_map[kind]] += 1
                if kind == 'Cast':
                    count_expr(value.value)
                else:
                    count_expr(value.a)
                    count_expr(value.b)
            elif isinstance(value, tvm.ir.Call):
                name = value.op.name.split('.')[-1]
                if name == 'if_then_else':
                    counts['simd.identity'] += 1
                elif name in ('exp', 'exp2', 'log', 'sqrt', 'rsqrt'):
                    counts['sfu.exp' if name == 'exp2' else 'sfu.' + name] += 1
                elif name in ('abs', 'fabs'):
                    counts['simd.max'] += 1
                else:
                    raise ValueError('unsupported scalar call ' + name)
                for arg in value.args:
                    if not isinstance(arg, tvm.ir.Var) and type(arg).__name__ not in ('IntImm', 'FloatImm'):
                        count_expr(arg)
        count_expr(expression, fma)
        if red:
            counts['simd.fma' if fma else 'simd.add'] += 1
        if not counts:
            counts['simd.identity'] = 1
        first_op = len(graph.operations)
        stage_stores = []
        max_regs = 0
        for flat in range(0, total, tile_n):
            cta = graph.cta(warps, 0, 128 * warps)
            if flat == 0 and completed:
                # One completion join represents the stage/kernel boundary;
                # sharing it avoids a quadratic all-to-all edge expansion.
                completed = [graph.add(cta, 0, 'barrier.registers', (0, 0, 1, 1),
                                       completed, stage=block.name_hint)]
            register_bytes = warps * 32 * 24 * 4
            for warp in range(warps):
                start = flat + warp * (tile_n // warps)
                stop = min(start + tile_n // warps, flat + tile_n, total)
                # Split only at a logical row boundary; all subtiles are in
                # the same TVM block/warp, including fused spatial axes.
                while start < stop:
                    row, column = divmod(start, columns)
                    extent = min(stop - start, columns - column)
                    prefix = np.unravel_index(row, shape[:-1]) if len(shape) > 1 else ()
                    domain = {v.var: (int(value), int(value)) for v, value in zip(spatial[:-1], prefix)}
                    domain[spatial[-1].var] = (column, column + extent - 1)
                    previous = graph.add(cta, warp, 'simd.fill', (0, 0, 1, extent), completed, stage=block.name_hint)
                    ring = []
                    maximum_load_bytes = 0
                    for kk in range(0, reduction, tile_k):
                        live_k = min(tile_k, reduction - kk)
                        analyzer = tvm.sym.Analyzer()
                        for variable, (lo, hi) in domain.items():
                            analyzer.update(variable, tvm.sym.ConstIntBound(lo, hi), override=True)
                        if red:
                            analyzer.update(red[0].var, tvm.sym.ConstIntBound(kk, kk + live_k - 1), override=True)
                        dependencies = completed if kk // tile_k < depth else [ring[kk // tile_k - depth]]
                        reads = []
                        loaded_bytes = 0
                        regions = set()
                        for load in loads:
                            source = load.source
                            dims = tuple(int(n) for n in source.shape)
                            indices = list(load.indices)
                            flattened = sum(index * math.prod(dims[i + 1:-1]) for i, index in enumerate(indices[:-1])) if len(dims) > 1 else tvm.tirx.const(0, 'int32')
                            row_bound = analyzer.const_int_bound(flattened)
                            col_bound = analyzer.const_int_bound(indices[-1])
                            # TE conditional expressions can mention the
                            # inactive next-state history element. Clipping
                            # bounds retains exactly the addressable superset.
                            rr = (max(0, int(row_bound.min_value)), min(math.prod(dims[:-1]) - 1, int(row_bound.max_value)))
                            cc = (max(0, int(col_bound.min_value)), min(dims[-1] - 1, int(col_bound.max_value)))
                            if rr[1] < rr[0] or cc[1] < cc[0]:
                                continue
                            region = (rr[0], cc[0], rr[1] - rr[0] + 1, cc[1] - cc[0] + 1)
                            tid = graph.tensor_ids[str(source.name)]
                            if (tid, region) in regions:
                                continue
                            regions.add((tid, region))
                            loaded_bytes += region[2] * region[3] * (4 if str(source.dtype) == 'float32' else 2)
                            reads.append(graph.add(cta, warp, 'ld.dram2reg', region, dependencies,
                                                   tid, source.dtype, block.name_hint))
                        maximum_load_bytes = max(maximum_load_bytes, loaded_bytes)
                        current = graph.add(cta, warp, 'barrier.registers', (0, 0, 1, 1), [previous, *reads], stage=block.name_hint)
                        for operation, count in counts.items():
                            for _ in range(count):
                                current = graph.add(cta, warp, operation, (0, 0, live_k, extent), [current], stage=block.name_hint)
                        previous = current
                        ring.append(current)
                    if red:
                        for _ in range(5):
                            previous = graph.add(cta, warp, 'shfl', (0, 0, 1, min(32, max(extent, reduction))), [previous], stage=block.name_hint)
                    register_bytes += maximum_load_bytes * min(depth, math.ceil(reduction / tile_k)) + 4 * extent
                    stage_stores.append(graph.add(cta, warp, 'st.reg2dram', (row, column, 1, extent),
                                                  [previous], graph.tensor_ids[str(output.name)], output.dtype, block.name_hint))
                    start += extent
            max_regs = max(max_regs, register_bytes)
            if register_bytes > min(hardware['register_file_per_sm_bytes'], warps * 32 * 255 * 4):
                raise ValueError('actual vector register ring exceeds capacity')
            old = graph.ctas[cta]
            graph.ctas[cta] = (*old[:4], register_bytes)
        completed = stage_stores
        report.append(dict(name=block.name_hint, output=str(output.name), shape=shape,
                           tile_n=tile_n, tile_k=tile_k, warps=warps, stages=depth,
                           operations=len(graph.operations) - first_op,
                           maximum_cta_register_bytes=max_regs,
                           arithmetic_ops=dict(counts)))
    return graph, dict(stages=report, nodes=len(graph.operations), ctas=len(graph.ctas),
                       tensor_tiers={name: graph.tensors[ident][-1] for name, ident in graph.tensor_ids.items()},
                       scope='Full staged TensorIR; SIMD vector arithmetic, global stage boundaries, register-ring pipeline')


def tvm_compatible_equal(tvm, a, b):
    import tvm_ffi
    return tvm_ffi.structural_equal(a, b)
