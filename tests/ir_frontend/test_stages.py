import unittest
from dataclasses import replace
from gtsim_frontend.ir.common import Event, SlotHandle, Tensor, TensorRegion, VerificationError
from gtsim_frontend.ir.flow import FlowKernel, FlowOp
from gtsim_frontend.ir.mapping import Mapping, Executor, Buffer
from gtsim_frontend.ir.arch import ArchKernel, ArchInstruction, CTA
from gtsim_frontend.ir.execution import ExecKernel, ExecProgram
from gtsim_frontend.programs.gemm_single import build
from gtsim_frontend.serialization.json_codec import dumps, loads


class StageTests(unittest.TestCase):
    def flow(self):
        slot=SlotHandle('stage',0,0)
        ops=(FlowOp('acquire','acquire','loader',outputs=(Event('owned','acquire','ready'),),slot=slot),
             FlowOp('load','load_async','loader',reads=(TensorRegion('A',(0,0),(128,64)),),inputs=('owned',),outputs=(Event('ready','load','ready'),),slot=slot),
             FlowOp('mma','matmul','consumer',inputs=('ready',),outputs=(Event('done','mma','complete'),),slot=slot),
             FlowOp('release','release','loader',inputs=('done',),outputs=(Event('released','release','released'),),slot=slot))
        return FlowKernel(symbol='flow',schedule_id='ref_h100_gemm_v0_1_0',tensors=(Tensor('A',(128,64)),),
                          executors=(Executor('loader','producer',(8,)),Executor('consumer','matrix',(0,1,2,3))),
                          buffers=(Buffer('stage',2,16384),),
                          mapping=Mapping(execute=(('loader','TMA'),('consumer','WGMMA')),store=(('stage','SMEM'),)),ops=ops)

    def test_all_stage_round_trips(self):
        arch=ArchKernel(symbol='native',ctas=(CTA(0,0,12,163840,180224),),
                        ops=(ArchInstruction(0,'load','cp.dram2sram_tma','LD',0,8),))
        p=build()
        program=ExecProgram(**vars(p),kernel_files=(('gemm0','kernel.gtsir'),))
        for obj in (self.flow(),arch,ExecKernel(**vars(arch)),program):
            self.assertEqual(obj,loads(dumps(obj)))

    def test_slot_lifetime_and_mapping(self):
        f=self.flow()
        for bad in (replace(f,ops=f.ops[:-1]),replace(f,ops=f.ops[:2]+(replace(f.ops[2],inputs=('absent',)),)+f.ops[3:]),
                    replace(f,mapping=replace(f.mapping,routes=(('HBF','SMEM','bulk_copy'),))),
                    replace(f,ops=(replace(f.ops[0],slot=SlotHandle('stage',0,1)),)+f.ops[1:]),
                    replace(f,ops=(f.ops[0],replace(f.ops[0],id='overwrite'))+f.ops[1:])):
            with self.assertRaises(VerificationError):bad.verify()

    def test_native_rejects_cycles_and_unknown_ops(self):
        a=ArchInstruction(0,'a','local_max','SIMD',0,0,after_complete=(1,))
        b=ArchInstruction(1,'b','local_sum','SIMD',0,0,after_issue=(0,))
        k=ArchKernel(symbol='bad',ctas=(CTA(0,0,12,163840,180224),),ops=(a,b))
        with self.assertRaises(VerificationError):k.verify()
        with self.assertRaises(VerificationError):replace(k,ops=(replace(a,op='tcgen05',after_complete=()),)).verify()


if __name__=='__main__':unittest.main()
