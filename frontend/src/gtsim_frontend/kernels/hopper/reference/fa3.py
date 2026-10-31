"""Independent full-Q-tile streaming-softmax Flow schedule, timing only."""
from ....ir.common import Tensor, TensorRegion
from ....ir.flow import FlowKernel
from ....ir.mapping import Mapping, Executor, Buffer
from ....serialization.json_codec import digest
from .builder import FlowBuilder


def instantiate(algorithm, num_sms=132):
    algorithm.verify()
    if algorithm.family!='fa3':raise ValueError('FA3 schedule requires FA3 algorithm')
    if type(num_sms) is not int or num_sms<1:raise ValueError('invalid SM count')
    batch,hq,hkv,q,kv=algorithm.dimensions
    builder=FlowBuilder()
    tensors=[];executors=[];buffers=[];execute=[];store=[]
    for b in range(batch):
        for h in range(hq):
            tensors.extend((Tensor(f'Q.{b}.{h}',(q,128),read_only=True),Tensor(f'O.{b}.{h}',(q,128))))
        for h in range(hkv):
            tensors.extend((Tensor(f'K.{b}.{h}',(128,kv),read_only=True),Tensor(f'V.{b}.{h}',(128,kv),read_only=True)))
    cta=0
    for b in range(batch):
        for h in range(hq):
            kv_head=h//(hq//hkv)
            for row in range(q//128):
                prefix=f'cta{cta}'
                producer=prefix+'.producer';matrix=prefix+'.matrix';vector=prefix+'.vector'
                builder.producer=producer
                executors.extend((Executor(producer,'producer',(8,)),Executor(matrix,'matrix',tuple(range(8))),Executor(vector,'vector',tuple(range(8)))))
                execute.extend(((producer,'TMA'),(matrix,'WGMMA'),(vector,'SIMD')))
                for name,size,slots in (('Q',32768,1),('K',32768,2),('V',32768,2)):
                    buffers.append(Buffer(prefix+'.'+name,slots,size));store.append((prefix+'.'+name,'SMEM'))
                qslot,owned=builder.acquire(prefix+'.Q.acquire',prefix+'.Q',0)
                qe,qvalue=builder.emit(prefix+'.Q.load','load_async',producer,inputs=(owned,),slot=qslot,
                    reads=(TensorRegion(f'Q.{b}.{h}',(row*128,0),(128,128)),),shape=(128,128),dtype='fp16')
                me,mvalue=builder.emit(prefix+'.m.init','compute',vector,semantic='initial_max_negative_infinity',shape=(128,))
                le,lvalue=builder.emit(prefix+'.l.init','compute',vector,semantic='initial_sum_zero',shape=(128,))
                oe,ovalue=builder.emit(prefix+'.O.init','compute',vector,semantic='initial_output_zero',shape=(128,128))
                q_consumers=[]
                for iteration in range(kv//128):
                    p=f'{prefix}.k{iteration}'
                    ks,ko=builder.acquire(p+'.K.acquire',prefix+'.K',iteration%2)
                    vs,vo=builder.acquire(p+'.V.acquire',prefix+'.V',iteration%2)
                    ke,kvalue=builder.emit(p+'.K.load','load_async',producer,inputs=(ko,),slot=ks,
                        reads=(TensorRegion(f'K.{b}.{kv_head}',(0,iteration*128),(128,128)),),shape=(128,128),dtype='fp16')
                    ve,vvalue=builder.emit(p+'.V.load','load_async',producer,inputs=(vo,),slot=vs,
                        reads=(TensorRegion(f'V.{b}.{kv_head}',(0,iteration*128),(128,128)),),shape=(128,128),dtype='fp16')
                    se,score=builder.emit(p+'.QK','matmul',matrix,inputs=(qe,ke),slot=ks,
                        operands=(qvalue,kvalue),semantic='scaled_QK',shape=(128,128))
                    q_consumers.append(se)
                    builder.release(p+'.K.release',ks,(se,))
                    maxima=[];max_events=[]
                    for branch in range(2):
                        e,v=builder.emit(f'{p}.max{branch}.local','compute',vector,inputs=(se,),operands=(score,),semantic='local_max',shape=(128,))
                        e,v=builder.emit(f'{p}.max{branch}.shuffle','compute',vector,inputs=(e,),operands=(v,),semantic='shfl_max',shape=(128,))
                        e,v=builder.emit(f'{p}.max{branch}.reduce','compute',vector,inputs=(e,),operands=(v,),semantic='reduce_max',shape=(128,))
                        max_events.append(e);maxima.append(v)
                    previous_me,previous_m=me,mvalue
                    me,mvalue=builder.emit(p+'.max.merge','compute',vector,inputs=tuple(max_events)+(me,),operands=tuple(maxima)+(mvalue,),semantic='merge_max',shape=(128,))
                    scale_event,scale=builder.emit(p+'.scale','compute',vector,inputs=(me,previous_me),operands=(mvalue,previous_m),semantic='scores_scale',shape=(128,2))
                    sums=[];sum_events=[];probabilities=[];prob_events=[]
                    for branch in range(2):
                        e,v=builder.emit(f'{p}.exp{branch}','compute',vector,inputs=(scale_event,se,me),operands=(score,mvalue),semantic='exp2',shape=(128,64))
                        prob_events.append(e);probabilities.append(v)
                        e,v=builder.emit(f'{p}.sum{branch}.local','compute',vector,inputs=(e,),operands=(v,),semantic='local_sum',shape=(128,))
                        e,v=builder.emit(f'{p}.sum{branch}.shuffle','compute',vector,inputs=(e,),operands=(v,),semantic='shfl_sum',shape=(128,))
                        e,v=builder.emit(f'{p}.sum{branch}.reduce','compute',vector,inputs=(e,),operands=(v,),semantic='reduce_sum',shape=(128,))
                        sum_events.append(e);sums.append(v)
                    sum_event,sum_value=builder.emit(p+'.sum.merge','compute',vector,inputs=tuple(sum_events),operands=tuple(sums),semantic='merge_sum',shape=(128,))
                    le,lvalue=builder.emit(p+'.logsum','compute',vector,inputs=(sum_event,scale_event,le),operands=(sum_value,scale,lvalue),semantic='logsum',shape=(128,))
                    oe,ovalue=builder.emit(p+'.O.scale','compute',vector,inputs=(scale_event,oe),operands=(scale,ovalue),semantic='acc_o_scale',shape=(128,128))
                    pe,pvalue=builder.emit(p+'.P.cast','compute',vector,inputs=tuple(prob_events),operands=tuple(probabilities),semantic='acc_s_cast',shape=(128,128),dtype='fp16')
                    oe,ovalue=builder.emit(p+'.PV','matmul',matrix,inputs=(ve,pe,oe,le),slot=vs,operands=(vvalue,pvalue,ovalue),semantic='PV_accumulate',shape=(128,128))
                    builder.release(p+'.V.release',vs,(oe,))
                builder.release(prefix+'.Q.release',qslot,tuple(q_consumers))
                oe,ovalue=builder.emit(prefix+'.normalize','compute',vector,inputs=(oe,le),operands=(ovalue,lvalue),semantic='acc_o_div',shape=(128,128))
                builder.emit(prefix+'.store','store_async',producer,inputs=(oe,),operands=(ovalue,),
                    writes=(TensorRegion(f'O.{b}.{h}',(row*128,0),(128,128)),),semantic='cast_fp16_store')
                cta+=1
    flow=FlowKernel(symbol=algorithm.symbol,source_hash=digest(algorithm),family='fa3',schedule_id='ref_h100_fa3_v0_1_0',
        tensors=tuple(tensors),executors=tuple(executors),buffers=tuple(buffers),
        mapping=Mapping(ctas=tuple((c,c%num_sms) for c in range(cta)),execute=tuple(execute),store=tuple(store)),ops=tuple(builder.ops))
    flow.verify()
    return flow
