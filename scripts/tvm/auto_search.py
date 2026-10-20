"""TVM MetaSchedule search, with GTSim builders/runners and cycle feedback."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import random
import traceback
import shutil
import time

import numpy as np
import tvm
from tvm.s_tir import meta_schedule as ms
from tvm.ir.utils import derived_object

from auto_ir import trace_json
from auto_rules import apply_rule, design_space, trace_choices, validate_choices
from gemm import ARCH, MEMORY, cycles, memory_statistics, run_simulation
from hopper_compiler import lower_hopper
from tier_memory import lower_module


def write_json(path,data):
    Path(path).write_text(json.dumps(data,indent=2)+'\n')


def search(mod,dimensions,hardware,build,out,trials=32,seed=0,jobs=4,memory=MEMORY,resume=False):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    previous=[]
    if (out/'search-history.json').exists():
        if not resume:raise ValueError('use a new output directory, or --resume')
        previous=json.loads((out/'search-history.json').read_text())
        archive=out/'resume-archives'/str(time.time_ns());archive.mkdir(parents=True)
        for pattern in ('database_*.json','cost-model.json','search-history.json'):
            for path in out.glob(pattern):shutil.move(str(path),archive/path.name)
    frequency_lines=[line.split(':',1)[1] for line in ARCH.read_text().splitlines() if line.startswith('core_frequency_mhz:')]
    hz=float(frequency_lines[0])*1e6
    history=[];observations=[];updates=[];build_counter=[0];pruning={}
    space=design_space(tvm,dimensions,hardware)
    strategies=space.get('prefetch',[0])
    # Balance a small native-search warmup before fitting. Otherwise early
    # regression on one observation can exclude its other memory strategy.
    # MetaSchedule still generates and mutates all kernel factors.
    strategy_warmup=min(4,trials//(2*len(strategies))) if len(strategies)>1 else 0

    def features(choices):
        bm,bn,bk,depth,groups,persistent=(choices['bm'],choices['n_per_group']*choices['groups'],choices['bk'],
                                        choices['pipeline_delay']+1,choices['groups'],choices['persistent'])
        raw=np.array([np.log2(bm)/8,np.log2(bn)/8,np.log2(bk)/6,depth/8,groups/2,persistent,choices.get('prefetch',0)],dtype='float64')
        return np.r_[1,raw,raw*raw,[raw[i]*raw[j] for i in range(len(raw)) for j in range(i)]]

    @derived_object
    class FeedbackModel(ms.cost_model.PyCostModel):
        def load(self,path):
            observations[:]=json.loads(Path(path).read_text())['observations']
        def save(self,path):
            write_json(path,dict(observations=observations,updates=updates,kind='ridge_log_simulated_seconds'))
        def update(self,context,candidates,results):
            accepted=[]
            for candidate,result in zip(candidates,results):
                if result.error_msg is None and result.run_secs:
                    choice=trace_choices(candidate.sch.trace)
                    seconds=float(np.mean([float(value) for value in result.run_secs]))
                    observations.append(dict(choices=choice,seconds=seconds))
                    accepted.append(dict(choices=choice,cycles=round(seconds*hz)))
            updates.append(dict(round=len(updates),observations=len(observations),accepted=accepted))
            self.save(str(out/'cost-model.json'))
        def predict(self,context,candidates):
            if strategy_warmup:
                counts={strategy:sum(row['choices'].get('prefetch',0)==strategy for row in observations)
                        for strategy in strategies}
                if min(counts.values())<strategy_warmup:
                    return np.array([1.0/(1+counts[trace_choices(candidate.sch.trace).get('prefetch',0)])
                                     for candidate in candidates],dtype='float64')
            x=np.array([features(trace_choices(candidate.sch.trace)) for candidate in candidates])
            if not observations:
                # Native evolutionary search samples an initial population;
                # scores are neutral until actual simulator feedback arrives.
                return np.ones(len(candidates),dtype='float64')
            train=np.array([features(row['choices']) for row in observations])
            y=np.log([row['seconds'] for row in observations])
            penalty=np.eye(train.shape[1])*.01;penalty[0,0]=1e-8
            gram=np.einsum('ni,nj->ij',train,train,optimize=False)
            rhs=np.einsum('ni,n->i',train,y,optimize=False)
            weights=np.linalg.solve(gram+penalty,rhs)
            prediction=np.einsum('ni,i->n',x,weights,optimize=False)
            if not np.isfinite(prediction).all():
                raise ValueError('nonfinite learned performance predictions')
            novelty=np.min(np.sum((x[:,None,:]-train[None,:,:])**2,axis=2),axis=1)
            return np.exp(-prediction)*np.exp(.03*np.minimum(novelty,2))

    @derived_object
    class Legal(ms.postproc.PyPostproc):
        def _initialize_with_tune_context(self,context):pass
        def apply(self,sch):
            try:validate_choices(trace_choices(sch.trace),dimensions,hardware);return True
            except (ValueError,KeyError) as error:
                reason=str(error)
                pruning[reason]=pruning.get(reason,0)+1
                return False
        def clone(self):return Legal()

    @derived_object
    class MutateDecision(ms.mutator.PyMutator):
        def _initialize_with_tune_context(self,context):pass
        def apply(self,trace,rand_state):
            rng=random.Random(int(rand_state))
            choices=[inst for inst in trace.insts if inst.kind.name=='SampleCategorical' and len(inst.attrs[0])>1]
            if not choices:return None
            inst=rng.choice(choices);previous=int(trace.decisions[inst]);options=list(range(len(inst.attrs[0])));options.remove(previous)
            return trace.with_decision(inst,rng.choice(options),True)
        def clone(self):return MutateDecision()

    @derived_object
    class Builder(ms.builder.PyBuilder):
        def build(self,inputs):
            result=[]
            for item in inputs:
                ident=build_counter[0];build_counter[0]+=1
                directory=out/f'candidate-{ident:04d}'
                scheduled=tvm.ir.save_json(item.mod)
                cached=next((row for row in previous if row['candidate']==ident and row['status']=='complete'),None)
                if cached is not None and (directory/'scheduled.json').read_text()==scheduled:
                    row=dict(cached,training_samples_before=len(observations),round=len(updates),reused=True)
                    history.append(row);result.append(ms.builder.BuilderResult(str(directory),None));continue
                if directory.exists():
                    archive=out/'resume-archives'/str(time.time_ns());archive.mkdir(parents=True)
                    shutil.move(str(directory),archive/directory.name)
                directory.mkdir()
                (directory/'scheduled.json').write_text(tvm.ir.save_json(item.mod))
                (directory/'scheduled.py').write_text(item.mod.script())
                row=dict(candidate=ident,directory=str(directory),training_samples_before=len(observations),round=len(updates))
                try:
                    native,pipeline,metadata=lower_hopper(tvm,item.mod,hardware)
                    # The general DAG lowerer consumes a separately serialized
                    # native module, never the schedule's choices or TE graph.
                    serialized=tvm.ir.save_json(native)
                    (directory/'native.json').write_text(serialized)
                    (directory/'native.py').write_text(native.script())
                    (directory/'pipeline.py').write_text(pipeline.script())
                    lowered=native
                    for compiler_pass in (tvm.tirx.transform.FlattenBuffer(),tvm.tirx.transform.LowerIntrin()):
                        lowered=compiler_pass(lowered)
                    (directory/'lowered-tvm.py').write_text(lowered.script())
                    native=tvm.ir.load_json(serialized)
                    ir,evidence=lower_module(tvm,native,hardware)
                    path=directory/'kernel.gtsir';ir.write(path)
                    metadata.update(evidence)
                    metadata.update(tir_sha256=hashlib.sha256(serialized.encode()).hexdigest(),
                                    ir_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                    write_json(directory/'lowering.json',metadata)
                    row.update(status='built',**metadata)
                    result.append(ms.builder.BuilderResult(str(directory),None))
                except Exception as error:
                    (directory/'error.log').write_text(traceback.format_exc())
                    row.update(status='build_failed',error=str(error))
                    result.append(ms.builder.BuilderResult(None,str(error)))
                history.append(row)
                write_json(out/'search-history.json',history)
            return result

    @derived_object
    class Runner(ms.runner.PyRunner):
        def run(self,inputs):
            def evaluate(item):
                directory=Path(item.artifact_path)
                row=next(row for row in history if row['directory']==str(directory))
                if row.get('reused'):
                    return ms.runner.RunnerResult([row['cycles']/hz],None)
                try:
                    command=run_simulation(build,ARCH,memory,directory/'kernel.gtsir',directory/'simulation.log')
                    count=cycles((directory/'simulation.log').read_text())
                    row.update(status='complete',cycles=count,command=command,
                               memory_statistics=memory_statistics((directory/'simulation.log').read_text()))
                    return ms.runner.RunnerResult([count/hz],None)
                except Exception as error:
                    row.update(status='run_failed',error=str(error))
                    return ms.runner.RunnerResult(None,str(error))
            with ThreadPoolExecutor(max_workers=jobs) as pool:
                values=list(pool.map(evaluate,inputs))
            write_json(out/'search-history.json',history)
            return [ms.runner.RunnerFuture(lambda:True,lambda value=value:value) for value in values]

    @derived_object
    class RecordTrace(ms.measure_callback.PyMeasureCallback):
        def apply(self,scheduler,task_id,candidates,built,results):
            for candidate,artifact,result in zip(candidates,built,results):
                if artifact.artifact_path:
                    directory=Path(artifact.artifact_path)
                    write_json(directory/'trace.json',trace_json(candidate.sch.trace))
                    (directory/'trace.py').write_text('\n'.join(map(str,candidate.sch.trace.as_python())))
                    row=next(row for row in history if row['directory']==str(directory))
                    row['choices']=trace_choices(candidate.sch.trace)
            write_json(out/'search-history.json',history)

    # Separate native design spaces for the supported warpgroup bindings.
    # All tiling/pipeline choices inside each space remain sampled trace RVs.
    def generate_spaces(sch):
        return [apply_rule(tvm,sch.copy(),dimensions,hardware,group_count=group) for group in space['groups']]
    generator=ms.space_generator.ScheduleFn(generate_spaces,
        sch_rules=[],postprocs=[Legal()],mutator_probs={MutateDecision():1.0})
    strategy=ms.search_strategy.EvolutionarySearch(population_size=64,init_min_unmeasured=32,
        genetic_num_iters=4,eps_greedy=.15)
    target=tvm.target.Target(dict(kind='cuda',arch='sm_90a',max_num_threads=1024,thread_warp_size=32))
    context=ms.TuneContext(mod=mod,target=target,space_generator=generator,search_strategy=strategy,
                          task_name='h100_fp16_gemm_gtsim',rand_state=seed+1,num_threads=1)
    model=FeedbackModel()
    write_json(out/'space.json',space)
    (out/'computation.json').write_text(tvm.ir.save_json(mod))
    (out/'computation.py').write_text(mod.script())
    database=ms.tune_tasks(tasks=[context],task_weights=[1.0],work_dir=str(out),
        max_trials_global=trials,num_trials_per_iter=min(jobs,trials),builder=Builder(),runner=Runner(),
        cost_model=model,database='json',task_scheduler='round-robin',
        measure_callbacks=[RecordTrace(),ms.measure_callback.AddToDatabase(),ms.measure_callback.UpdateCostModel()])
    model.save(str(out/'cost-model.json'))
    write_json(out/'postproc-pruning.json',pruning)
    complete=[row for row in history if row['status']=='complete']
    coverage={strategy:sum(row['choices'].get('prefetch',0)==strategy for row in complete)
              for strategy in strategies}
    write_json(out/'memory-strategy-coverage.json',dict(legal_strategies=strategies,
               minimum_per_strategy=strategy_warmup,completed=coverage))
    if any(count<strategy_warmup for count in coverage.values()):
        raise RuntimeError('native search did not cover every legal memory strategy')
    if len(complete)<2 or len({row['tir_sha256'] for row in complete})<2:
        raise RuntimeError('search did not produce multiple distinct successfully simulated kernels')
    if len(updates)<2 or not any(row['training_samples_before'] for row in complete):
        raise RuntimeError('search did not exercise simulated-performance feedback')
    best=min(complete,key=lambda row:row['cycles'])
    record=database.query_tuning_record(mod,target,'main')
    if record is None:raise RuntimeError('native MetaSchedule database has no best record')
    return best,history,updates,database
