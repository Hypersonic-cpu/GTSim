"""Independent bounded TIR trace audit for persistent-GEMM lowering tests.

Uses TVM substitution/simplification, not the lowerer's expression evaluator or
C++ builder. Checks completion-event reachability separately from issue events.
This is a synchronization/region audit, not execution of floating-point GEMM.
"""

from collections import defaultdict
from functools import cache
import re


def check_semantics(tvm, func, ir, selected_ctas=None):
    t = tvm.tirx
    analyzer = tvm.sym.Analyzer()

    def value(expr, env):
        for var, bound in env.items():
            analyzer.bind(var, bound, allow_override=True)
        result = analyzer.simplify(expr)
        binary = {"BitwiseOr": lambda a, b: a | b, "BitwiseAnd": lambda a, b: a & b}
        if type(result).__name__ in binary:
            return binary[type(result).__name__](value(result.a, {}), value(result.b, {}))
        if not isinstance(result, t.IntImm):
            raise AssertionError(f"nonconstant trace expression: {type(result).__name__} {getattr(result, 'op', None)} {result}")
        return int(result)

    def trace(stmt, env, tx, cta):
        if isinstance(stmt, t.SeqStmt):
            for child in stmt.seq:
                yield from trace(child, env, tx, cta)
        elif isinstance(stmt, t.For):
            # Scalar accumulator/staging writes have no synchronization calls.
            if isinstance(stmt.body, t.BufferStore):
                return
            tag = stmt.thread_binding.thread_tag if stmt.thread_binding is not None else None
            indices = [tx] if tag == "threadIdx.x" else [cta] if tag == "blockIdx.x" else range(
                value(stmt.min, env), value(stmt.min + stmt.extent, env))
            for index in indices:
                yield from trace(stmt.body, env | {stmt.loop_var: t.IntImm("int32", index)}, tx, cta)
        elif isinstance(stmt, t.IfThenElse):
            body = stmt.then_case if value(stmt.condition, env) else stmt.else_case
            if body is not None:
                yield from trace(body, env, tx, cta)
        elif isinstance(stmt, t.Evaluate) and isinstance(stmt.value, tvm.ir.Call):
            if isinstance(stmt.value.op, tvm.ir.Op):
                yield stmt.value, env

    def address(expr):
        while isinstance(expr, tvm.ir.Call):
            expr = expr.args[0]
        assert isinstance(expr, tvm.ir.TensorLoad), f"unexpected address: {expr}"
        return expr

    ctacount = len(ir.ctas)
    selected_ctas = sorted(set(selected_ctas if selected_ctas is not None else (0, ctacount - 1)))
    bn = int(next(node.buffer.shape[1] for node in walk(func.body)
                  if isinstance(node, t.AllocBuffer) and str(node.buffer.name) == 'stage_b')) * 64
    totals = defaultdict(int)

    def op_name(call):
        return call.op.name

    for cta in selected_ctas:
        assert ir.ctas[cta][0:2] == (cta, cta), 'persistent CTA/SM assignment changed'
        ids = [i for i, op in enumerate(ir.operations) if op.cta == cta]
        # Four fragments execute as one WGMMA collective in the simulator.
        groups = defaultdict(list)
        for ident in ids:
            op = ir.operations[ident]
            key = re.sub(r'_w\d+', f'_wg{op.warp // 4}', op.name) if op.op.startswith('wgmma.') else op.name
            groups[key].append(ident)
        canon = {ident: members[0] for members in groups.values() for ident in members}
        members = {items[0]: items for items in groups.values()}
        bits = {ident: 1 << index for index, ident in enumerate(members)}

        @cache
        def before(ident):
            result = 0
            for item in members[ident]:
                op = ir.operations[item]
                for parent in op.after_complete:
                    result |= bits[canon[parent]] | before(canon[parent])
                for parent in op.after_issue:
                    result |= before(canon[parent])
            return result

        def requires(sources, targets):
            required = 0
            for source in sources:
                required |= bits[canon[source]]
            for target in targets:
                assert before(canon[target]) & required == required, f'missing completion ordering at {ir.operations[target].name}'
                totals['completion_checks'] += 1

        actors = {tx: list(trace(func.body, {}, tx, cta)) for tx in [*range(0, bn, 128), 256]}
        maps = {str(call.args[0].name): call for call, _ in actors[256]
                if op_name(call) == 'tirx.tensormap_encode_tiled'}
        waves = defaultdict(list)
        for call, env in actors[256]:
            named = {str(var.name): int(val) for var, val in env.items()}
            if 'wave' in named:
                waves[named['wave']].append((call, env))
        previous_output, consumers, generations = [], [[], []], [0, 0]
        for wave, events in waves.items():
            tile = wave * ctacount + cta
            prefix = f'TB{cta}_Tile{tile}_'
            current = [ident for ident in ids if ir.operations[ident].name.startswith(prefix)]
            assert current, 'TIR tile missing from graph'
            graph_loads = [i for i in current if ir.operations[i].op == 'cp.dram2sram_tma']
            graph_output = [i for i in current if ir.operations[i].op == 'cp.sram2dram_tma']
            loads = [(call, env) for call, env in events if op_name(call) == 'tirx.ptx.cp_async_bulk_tensor_g2s_cluster']
            outputs = [(call, env) for call, env in events if op_name(call) == 'tirx.ptx.cp_async_bulk_tensor_s2g']
            assert len(loads) == len(graph_loads) and len(outputs) == len(graph_output), 'native/graph TMA count mismatch'
            for ident, (call, env) in zip(graph_loads + graph_output, loads + outputs):
                output = op_name(call).endswith('s2g')
                descriptor = maps[str(call.args[0 if output else 1].name)]
                offset = 1 if output else 2
                region = (value(call.args[offset + 1], env), value(call.args[offset], env),
                          int(descriptor.args[6]), int(descriptor.args[5]))
                assert ir.operations[ident].tile == region, 'native/graph TMA coordinates mismatch'
                assert ir.operations[ident].warp == 8, 'TMA producer warp changed'
                totals['region_checks'] += 1
            requires(previous_output, graph_loads)
            for previous, following in zip(graph_output, graph_output[1:]):
                assert previous in ir.operations[following].after_issue, 'TMA output issue order missing'
                totals['issue_checks'] += 1
            # The bulk wait without .read waits for full completion. Its CTA
            # join is the last event in EVERY participating actor's tile.
            waits = [index for index, (call, _) in enumerate(events) if op_name(call) == 'tirx.ptx.cp_async_bulk_wait_group']
            assert len(waits) == 1 and int(events[waits[0]][0].args[0]) == 0
            assert waits[0] < len(events) - 1 and op_name(events[-1][0]) == 'tirx.cuda.cta_sync'
            for tx, actor in actors.items():
                selected = [(call, env) for call, env in actor if any(str(var.name) == 'wave' and int(val) == wave for var, val in env.items())]
                assert sum(op_name(call) == 'tirx.cuda.cta_sync' for call, _ in selected) == 3
                if tx == 256:
                    continue
                mma = [i for i in current if ir.operations[i].op.startswith('wgmma.') and ir.operations[i].warp == tx // 32]
                native_mma = [(call, env) for call, env in selected if op_name(call) == 'tirx.ptx.wgmma_f16_ss']
                assert len(mma) == len(native_mma), 'native/graph WGMMA count mismatch'
                assert all(len(members[canon[i]]) == 4 for i in mma), 'incomplete warpgroup'
                assert sum(op_name(call) == 'tirx.ptx.wgmma_wait_group' and int(call.args[0]) == 0 for call, _ in selected) == len({re.search(r'_k(\d+)', ir.operations[i].name)[1] for i in mma})
            steps = defaultdict(list)
            for ident, (call, env) in zip(graph_loads, loads):
                step = next(int(val) for var, val in env.items() if str(var.name) == 'kk')
                steps[step].append(ident)
            previous_loads = []
            for step, stage_loads in steps.items():
                for ident in stage_loads:
                    assert set(previous_loads).issubset(ir.operations[ident].after_issue), 'TMA stage issue order missing'
                    totals['issue_checks'] += 1
                previous_loads = stage_loads
                call, env = loads[graph_loads.index(stage_loads[0])]
                slot = value(address(call.args[0]).indices[0], env)
                requires(consumers[slot], stage_loads)
                stage_mma = [i for i in current if ir.operations[i].op.startswith('wgmma.') and f'_k{step}_' in ir.operations[i].name]
                requires(stage_loads, stage_mma)
                for tx, actor in actors.items():
                    for wait, local in actor:
                        named = {str(var.name): int(val) for var, val in local.items()}
                        if named.get('wave') != wave or named.get('kk') != step or op_name(wait) != 'tirx.cuda.mbarrier_wait':
                            continue
                        barrier = address(wait.args[0])
                        assert value(barrier.indices[0], local) == slot
                        expected = generations[slot] - (str(barrier.source.name) == 'empty')
                        assert expected >= 0 and value(wait.args[1], local) == expected % 2, 'incorrect persistent mbarrier phase'
                        totals['phase_checks'] += 1
                consumers[slot] = stage_mma
                generations[slot] += 1
            staged = [i for i in current if ir.operations[i].op == 'st.reg2sram']
            requires([i for slot in consumers for i in slot if i in current], staged)
            requires(staged, graph_output)
            previous_output = graph_output
            totals['tiles_checked'] += 1
        totals['ctas_checked'] += 1
    return dict(totals)


def walk(stmt):
    """Small statement-only walker kept separate from production recognition."""
    if hasattr(stmt, 'seq'):
        for child in stmt.seq:
            yield from walk(child)
    if hasattr(stmt, 'body'):
        yield from walk(stmt.body)
    if hasattr(stmt, 'then_case'):
        yield from walk(stmt.then_case)
        if stmt.else_case is not None:
            yield from walk(stmt.else_case)
    yield stmt


def multiwave_cases(sms):
    """Odd/even, one/four K steps, both warpgroup counts, and partial tails."""
    columns = sms // 3 + 1
    cases = [((384, columns * 256, 64 * steps), (128, 256, 64)) for steps in (1, 2, 3, 4)]
    cases += [((64, (3 * sms + 4) * 128, 96), (64, 128, 32)),
              ((64, (sms + 3) * 256, 384), (64, 256, 128)),
              ((128, (sms + 3) * 128, 192), (128, 128, 64))]
    return cases
