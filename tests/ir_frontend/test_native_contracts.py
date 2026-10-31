import unittest
from dataclasses import replace
from gtsim_frontend.ir.common import VerificationError
from gtsim_frontend.programs.gemm_single import build
from gtsim_frontend.passes.instantiate import instantiate
from gtsim_frontend.passes.lower_hopper_gemm import lower
from gtsim_frontend.serialization.json_codec import loads,dumps
from gtsim_frontend.serialization.printer import print_ir


class NativeContractTests(unittest.TestCase):
    def test_collective_and_output_validation(self):
        flow=instantiate(build(128,256,128).launches[0].algorithm)
        arch=lower(flow)
        self.assertEqual(arch,loads(dumps(arch)))
        ops=list(arch.ops);i=next(i for i,o in enumerate(ops) if o.op=='wgmma.m64n64k16')
        ops[i]=replace(ops[i],op='local_max',pipeline='SIMD')
        with self.assertRaises(VerificationError):replace(arch,ops=tuple(ops)).verify()
        with self.assertRaises(VerificationError):replace(arch,outputs=()).verify()
        ops=list(arch.ops);ops[-1]=replace(ops[-1],tile=(0,0,64,64))
        with self.assertRaises(VerificationError):replace(arch,ops=tuple(ops)).verify()
        self.assertIn('arch.kernel @gemm',print_ir(arch))
        self.assertIn('map.cta',print_ir(flow))

    def test_unknown_flow_compute_and_incomplete_output(self):
        flow=instantiate(build(128,256,64).launches[0].algorithm)
        ops=list(flow.ops);ops[0]=replace(ops[0],semantic='unrecognized_compute')
        with self.assertRaises(ValueError):lower(replace(flow,ops=tuple(ops)))
        with self.assertRaises(VerificationError):replace(flow,ops=flow.ops[:-1]).verify()


if __name__=='__main__':unittest.main()
