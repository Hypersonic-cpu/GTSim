#!/usr/bin/env python3
"""Native MetaSchedule evolutionary serving search with measured GTSim feedback."""
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import shutil
import subprocess
import sys
import time
import traceback

from pinned_tvm import ROOT, TVM_COMMIT, load_tvm


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def search(tvm, mod, kind, hardware, architecture, build, out, trials, seed):
    import numpy as np
    from tvm.s_tir import meta_schedule as ms
    from tvm.ir.utils import derived_object
    from auto_ir import trace_json
    from gemm import cycles, memory_statistics
    from serve_rules import apply, choices, space
    from serve_lower import lower_vector
    from serve_fa2 import compile_online, lower_online
    out.mkdir(parents=True, exist_ok=False)
    history, observations, updates = [], [], []
    hz = hardware['core_frequency_mhz'] * 1e6
    counter = [0]
    keys = list(space(kind))

    def features(config):
        raw = np.array([math.log2(config[key] + 1) for key in keys]) / 8
        return np.r_[1, raw, raw * raw, [raw[i] * raw[j] for i in range(len(raw)) for j in range(i)]]

    @derived_object
    class Model(ms.cost_model.PyCostModel):
        def load(self, path):
            observations[:] = json.loads(Path(path).read_text())['observations']

        def save(self, path):
            write_json(path, dict(observations=observations, updates=updates,
                                  kind='ridge_log_simulated_seconds'))

        def update(self, context, candidates, results):
            for candidate, result in zip(candidates, results):
                if result.error_msg is None and result.run_secs:
                    observations.append(dict(choices=choices(candidate.sch.trace, kind),
                                             seconds=float(np.mean([float(value) for value in result.run_secs]))))
            updates.append(dict(round=len(updates), observations=len(observations)))
            self.save(out / 'cost-model.json')

        def predict(self, context, candidates):
            if not observations:
                return np.ones(len(candidates), dtype='float64')
            train = np.array([features(row['choices']) for row in observations])
            values = np.log([row['seconds'] for row in observations])
            penalty = np.eye(train.shape[1]) * .01
            penalty[0, 0] = 1e-8
            weights = np.linalg.solve(train.T @ train + penalty, train.T @ values)
            x = np.array([features(choices(candidate.sch.trace, kind)) for candidate in candidates])
            novelty = np.min(np.sum((x[:, None, :] - train[None, :, :]) ** 2, axis=2), axis=1)
            return np.exp(-(x @ weights)) * np.exp(.05 * np.minimum(novelty, 2))

    @derived_object
    class Legal(ms.postproc.PyPostproc):
        def _initialize_with_tune_context(self, context):
            pass

        def apply(self, sch):
            if kind == 'gdn_split':
                # Projection layouts are measured seeds. The builder still checks
                # actual resources after native rfactor and fusion transformations.
                return True
            c = choices(sch.trace, kind)
            if kind == 'fa2':
                params = {str(p.name): p for p in mod['main'].params}
                dim = int(params['Q'].shape[-1])
                group = int(params['Q'].shape[1]) // int(params['K'].shape[1])
                bm, bn, warps, depth = 16 * c['warps'], c['kv_tile'], c['warps'], c['stages']
                shared = 2 * dim * (bm + 2 * depth * bn)
                registers = 4 * bm * (dim + bn) + 2 * bm * dim + warps * 32 * 24 * 4
                return (group <= bm and shared <= min(hardware['shared_memory_per_cta_bytes'], hardware['shared_memory_per_sm_bytes'])
                        and registers <= min(hardware['register_file_per_sm_bytes'], warps * 32 * 255 * 4))
            if c['output_tile'] < 32 * c['warps']:
                return False
            registers = ((4 * c['reduction_tile'] * (c['output_tile'] + 4 * c['warps']) * c['stages']) if kind == 'gdn' else (2 * c['reduction_tile'] * (c['output_tile'] + c['warps']) * c['stages'])) + 4 * c['output_tile'] + c['warps'] * 32 * 24 * 4
            return registers <= min(hardware['register_file_per_sm_bytes'], c['warps'] * 32 * 255 * 4)

        def clone(self):
            return Legal()

    @derived_object
    class Mutator(ms.mutator.PyMutator):
        def _initialize_with_tune_context(self, context):
            pass

        def apply(self, trace, rand_state):
            rng = random.Random(int(rand_state))
            sampled = [inst for inst in trace.insts if inst.kind.name == 'SampleCategorical']
            selected = [inst for key, inst in zip(keys, sampled) if key != 'inline' and len(inst.attrs[0]) > 1]
            if not selected:
                return None
            inst = rng.choice(selected)
            options = [i for i in range(len(inst.attrs[0])) if i != int(trace.decisions[inst])]
            return trace.with_decision(inst, rng.choice(options), True)

        def clone(self):
            return Mutator()

    @derived_object
    class Builder(ms.builder.PyBuilder):
        def build(self, inputs):
            results = []
            for item in inputs:
                number = counter[0]
                counter[0] += 1
                directory = out / f'candidate-{number:04d}'
                directory.mkdir()
                row = dict(candidate=number, directory=str(directory),
                           training_samples_before=len(observations), status='building')
                history.append(row)
                started = time.monotonic()
                try:
                    serialized = tvm.ir.save_json(item.mod)
                    (directory / 'scheduled.json').write_text(serialized)
                    (directory / 'scheduled.tir').write_text(item.mod.script())
                    if kind == 'fa2':
                        native = compile_online(tvm, tvm.ir.load_json(serialized))
                        native_serialized = tvm.ir.save_json(native)
                        (directory / 'native.json').write_text(native_serialized)
                        (directory / 'native.tir').write_text(native.script())
                        graph, evidence = lower_online(tvm, tvm.ir.load_json(native_serialized), hardware)
                    else:
                        graph, evidence = lower_vector(tvm, tvm.ir.load_json(serialized), hardware)
                    graph.write(directory / 'kernel.gtsir')
                    row.update(status='built', nodes=len(graph.operations),
                               tir_sha256=hashlib.sha256(serialized.encode()).hexdigest(),
                               ir_sha256=hashlib.sha256((directory / 'kernel.gtsir').read_bytes()).hexdigest(),
                               build_wall_seconds=time.monotonic() - started)
                    write_json(directory / 'lowering.json', evidence)
                    del graph
                    gc.collect()
                    results.append(ms.builder.BuilderResult(str(directory), None))
                    print('BUILT', number, row['nodes'], 'nodes', flush=True)
                except Exception as error:
                    row.update(status='build_failed', error=str(error))
                    (directory / 'error.log').write_text(traceback.format_exc())
                    results.append(ms.builder.BuilderResult(None, str(error)))
                    print('BUILD REJECTED', number, str(error), flush=True)
                write_json(out / 'search-history.json', history)
            return results

    def measure(directory, row, suffix=''):
        log = directory / ('simulation' + suffix + '.log')
        resource = directory / ('resources' + suffix + '.txt')
        command = [str(build / 'gtsim_serve_ir'), str(architecture), str(directory / 'kernel.gtsir')]
        timed = ['/usr/bin/time', '-l' if platform.system() == 'Darwin' else '-v', '-o', str(resource), *command]
        started = time.monotonic()
        with log.open('w') as output:
            result = subprocess.run(timed, stdout=output, stderr=subprocess.STDOUT, check=False)
        if result.returncode:
            raise RuntimeError(f'simulator exit {result.returncode}; see {log}')
        text = log.read_text()
        count = cycles(text)
        match = re.search(r'Finished: (\d+)/(\d+)', text)
        if not match or match[1] != match[2]:
            raise RuntimeError('simulation did not drain')
        stats = memory_statistics(text)
        if stats['HBF']['write_bytes'] != 0 or (kind == 'fa2' and stats['HBF']['read_bytes'] != 0):
            raise RuntimeError('invalid learned-weight/state placement traffic')
        resource_text = resource.read_text()
        peaks = [int(n) for n in re.findall(r'(\d+)\s+(?:maximum resident set size|peak memory footprint)', resource_text)]
        row.update(cycles=count, us=count / hardware['core_frequency_mhz'], command=command,
                   simulation_wall_seconds=time.monotonic() - started,
                   memory_statistics=stats, maximum_process_memory_bytes=max(peaks, default=0),
                   stage_timings={name: dict(first=int(first), complete=int(last))
                                  for name, first, last in re.findall(r'Stage timing: (\S+) first (\d+) complete (\d+)', text)})
        return count

    @derived_object
    class Runner(ms.runner.PyRunner):
        def run(self, inputs):
            results = []
            # Serial by design: previous simulation peak was >3GB.
            for item in inputs:
                directory = Path(item.artifact_path)
                row = next(row for row in history if row['directory'] == str(directory))
                try:
                    count = measure(directory, row)
                    row['status'] = 'complete'
                    result = ms.runner.RunnerResult([count / hz], None)
                    print('MEASURED', row['candidate'], count, 'cycles', flush=True)
                except Exception as error:
                    row.update(status='run_failed', error=str(error))
                    result = ms.runner.RunnerResult(None, str(error))
                    print('RUN FAILED', row['candidate'], str(error), flush=True)
                results.append(ms.runner.RunnerFuture(lambda: True, lambda result=result: result))
                write_json(out / 'search-history.json', history)
            return results

    @derived_object
    class Record(ms.measure_callback.PyMeasureCallback):
        def apply(self, scheduler, task_id, candidates, artifacts, results):
            for candidate, artifact in zip(candidates, artifacts):
                if artifact.artifact_path:
                    directory = Path(artifact.artifact_path)
                    write_json(directory / 'trace.json', trace_json(candidate.sch.trace))
                    row = next(row for row in history if row['directory'] == str(directory))
                    row['choices'] = choices(candidate.sch.trace, kind)
            write_json(out / 'search-history.json', history)

    generator = ms.space_generator.ScheduleFn(
        lambda sch: [apply(tvm, sch.copy(), kind, dict(inline=level)) for level in space(kind).get('inline', [0])], sch_rules=[],
                                               postprocs=[Legal()], mutator_probs={Mutator(): 1.0})
    target = tvm.target.Target(dict(kind='cuda', arch='sm_80', max_num_threads=1024, thread_warp_size=32))
    context = ms.TuneContext(mod=mod, target=target, space_generator=generator,
                            search_strategy=ms.search_strategy.EvolutionarySearch(
                                population_size=32, init_min_unmeasured=16, genetic_num_iters=4, eps_greedy=.2),
                            task_name='a100_' + kind + '_gtsim', rand_state=seed + 1, num_threads=1)
    model = Model()
    (out / 'computation.json').write_text(tvm.ir.save_json(mod))
    (out / 'computation.tir').write_text(mod.script())
    write_json(out / 'space.json', space(kind))
    database = ms.tune_tasks(tasks=[context], task_weights=[1.0], work_dir=str(out),
                             max_trials_global=trials, num_trials_per_iter=1,
                             builder=Builder(), runner=Runner(), cost_model=model, database='json',
                             task_scheduler='round-robin', measure_callbacks=[Record(),
                                 ms.measure_callback.AddToDatabase(), ms.measure_callback.UpdateCostModel()])
    successful = [row for row in history if row['status'] == 'complete']
    if len(successful) < 2 or len({row['tir_sha256'] for row in successful}) < 2 or len(observations) < 2:
        raise RuntimeError('insufficient distinct measured schedules and feedback')
    best = min(successful, key=lambda row: row['cycles'])
    source = Path(best['directory'])
    replay = tvm.s_tir.Schedule(mod, debug_mask='all')
    tvm.s_tir.schedule.Trace.apply_json_to_schedule(json.loads((source / 'trace.json').read_text()), replay)
    tvm.ir.assert_structural_equal(replay.mod, tvm.ir.load_json((source / 'scheduled.json').read_text()))
    record = database.query_tuning_record(mod, target, 'main')
    if record is None:
        raise RuntimeError('MetaSchedule did not save a best record')
    confirmation = {}
    if measure(source, confirmation, '-confirm') != best['cycles']:
        raise RuntimeError('best schedule replay cycles differ')
    best_dir = out / 'best'
    best_dir.mkdir()
    for name in ['scheduled.json', 'scheduled.tir', 'native.json', 'native.tir', 'trace.json', 'lowering.json', 'simulation.log', 'resources.txt']:
        if (source / name).exists():
            shutil.copyfile(source / name, best_dir / name)
    result = dict(best=best, successful_candidates=len(successful), attempted_candidates=len(history),
                  feedback_updates=len(updates), workers=1, tvm_commit=TVM_COMMIT,
                  best_replay_confirmed=True, scope='simulated-cycle optimum among tested native MetaSchedule candidates')
    write_json(out / 'results.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=['fa2', 'gdn', 'projection'], required=True)
    parser.add_argument('--n', type=int, default=12288)
    parser.add_argument('--k', type=int, default=4096)
    parser.add_argument('--seq', type=int, default=32768)
    parser.add_argument('--trials', type=int, default=12)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--small', action='store_true')
    parser.add_argument('--architecture', type=Path, default=ROOT / 'configs/v0.1.0/a100-pcie40-3m2f-grade1-simple.yaml')
    parser.add_argument('--build-dir', type=Path, default=ROOT / 'build')
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'TVM_NUM_THREADS']:
        os.environ[key] = '1'
    tvm = load_tvm()
    sys.path.insert(0, str(ROOT / 'workloads/serve'))
    import semantics
    if args.kind == 'fa2':
        tensors = semantics.fa2(seq=args.seq, dim=64 if args.small else 256)
    elif args.kind == 'gdn':
        tensors = semantics.gdn(hidden=64, nk=2, nv=4, dk=16, dv=16, ff=128) if args.small else semantics.gdn()
    else:
        tensors = semantics.projection(args.n, args.k)
    mod = tvm.IRModule({'main': tvm.te.create_prim_func(tensors)})
    hardware = json.loads(subprocess.check_output([str(args.build_dir / 'gtsim_serve_ir'), str(args.architecture), '--describe-config'], text=True))
    from serve_rules import apply
    from serve_memory import annotate
    mod = annotate(tvm, mod, hardware['topology'])
    result = search(tvm, mod, args.kind, hardware, args.architecture.resolve(), args.build_dir.resolve(),
                    args.out_dir.resolve(), args.trials, args.seed)
    print(json.dumps(dict(best_cycles=result['best']['cycles'], best_choices=result['best']['choices'],
                          successful=result['successful_candidates'], feedback_updates=result['feedback_updates']), indent=2), flush=True)


if __name__ == '__main__':
    main()
