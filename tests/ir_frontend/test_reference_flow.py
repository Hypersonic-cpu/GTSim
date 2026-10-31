import unittest
from collections import Counter
from dataclasses import replace
from gtsim_frontend.ir.common import VerificationError
from gtsim_frontend.passes.instantiate import instantiate
from gtsim_frontend.programs.gemm_single import build as gemm
from gtsim_frontend.programs.fa3_single import build as fa3
from gtsim_frontend.serialization.json_codec import dumps,loads,digest


class ReferenceFlowTests(unittest.TestCase):
    def test_gemm_regions_and_odd_even_slot_reuse(self):
        for k in (64,128,192,256):
            f=instantiate(gemm(m=512,n=512,k=k).launches[0].algorithm,num_sms=2)
            stores=[r for o in f.ops for r in o.writes]
            self.assertEqual(len(stores),8)
            self.assertEqual({(r.offsets,r.shape) for r in stores},
                             {((m,n),(128,256)) for m in range(0,512,128) for n in (0,256)})
            self.assertEqual(sum(o.capability=='matmul' for o in f.ops),8*(k//64))
            self.assertEqual(len(f.buffers),2)
            self.assertEqual(loads(dumps(f)),f)
            self.assertEqual(digest(f),digest(instantiate(gemm(m=512,n=512,k=k).launches[0].algorithm,2)))
            for o in f.ops:
                for r in o.reads:
                    if r.tensor=='A':self.assertEqual(r.shape,(128,64))
                    if r.tensor=='B':self.assertEqual(r.shape,(64,256))

    def test_fa3_gqa_and_full_softmax_morphology(self):
        f=instantiate(fa3(heads_q=8,heads_kv=2,seq_q=256,seq_kv=384).launches[0].algorithm,2)
        counts=Counter(o.semantic for o in f.ops)
        rounds=8*2*3
        for semantic in ('scaled_QK','PV_accumulate','merge_max','scores_scale','merge_sum','logsum','acc_o_scale','acc_s_cast'):
            self.assertEqual(counts[semantic],rounds)
        for semantic in ('local_max','shfl_max','reduce_max','exp2','local_sum','shfl_sum','reduce_sum'):
            self.assertEqual(counts[semantic],2*rounds)
        self.assertEqual(counts['acc_o_div'],16)
        for cta in range(16):
            q_head=cta//2
            loads=[r.tensor for o in f.ops if o.id.startswith(f'cta{cta}.') for r in o.reads]
            self.assertEqual(set(loads),{f'Q.0.{q_head}',f'K.0.{q_head//4}',f'V.0.{q_head//4}'})
        stores=[r for o in f.ops for r in o.writes]
        self.assertEqual(len(stores),16)
        self.assertEqual(len({(r.tensor,r.offsets) for r in stores}),16)
        self.assertEqual(loads_fn(f),f)

    def test_dangling_value_rejected(self):
        f=instantiate(gemm(m=128,n=256,k=64).launches[0].algorithm)
        ops=list(f.ops)
        i=next(i for i,o in enumerate(ops) if o.capability=='matmul')
        ops[i]=replace(ops[i],operands=('unknown_tile',))
        with self.assertRaises(VerificationError):replace(f,ops=tuple(ops)).verify()


def loads_fn(f):return loads(dumps(f))


if __name__=='__main__':unittest.main()
