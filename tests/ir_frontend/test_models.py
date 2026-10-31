import json
import unittest
from dataclasses import replace
from gtsim_frontend.ir.common import VerificationError
from gtsim_frontend.ir.program import Command, Program
from gtsim_frontend.programs.gemm_single import build as gemm
from gtsim_frontend.programs.fa3_single import build as fa3
from gtsim_frontend.serialization.json_codec import dumps, loads, digest


class ModelTests(unittest.TestCase):
    def test_examples_round_trip_and_hash(self):
        for p in (gemm(), fa3(), fa3(heads_q=16, heads_kv=8, seq_q=512, seq_kv=1536)):
            self.assertEqual(p, loads(dumps(p)))
            self.assertEqual(digest(p), digest(loads(dumps(p))))
            self.assertEqual(
                p.launches[0].algorithm, loads(dumps(p.launches[0].algorithm))
            )

    def test_reference_shape_boundary(self):
        for build in (
            lambda: gemm(m=1),
            lambda: gemm(k=65),
            lambda: gemm(dtype="bf16"),
            lambda: fa3(seq_q=1),
            lambda: fa3(head_dim=256),
            lambda: fa3(heads_q=3, heads_kv=2),
        ):
            with self.assertRaises(VerificationError):
                build()

    def test_stream_tensor_event_contracts(self):
        p = gemm()
        for bad in (
            replace(p, streams=("a", "b")),
            replace(p, tensors=p.tensors[:-1]),
            replace(p, commands=(Command("WaitEvent", "missing"),) + p.commands),
            replace(
                p,
                launches=(
                    replace(
                        p.launches[0],
                        algorithm=replace(p.launches[0].algorithm, family="unknown"),
                    ),
                ),
            ),
            replace(p, tensors=(replace(p.tensors[0], shape=(1, 1)),) + p.tensors[1:]),
        ):
            with self.assertRaises(VerificationError):
                bad.verify()
        Program(symbol="empty").verify()
        replace(
            p,
            commands=p.commands
            + (Command("RecordEvent", "done"), Command("WaitEvent", "done")),
        ).verify()

    def test_codec_rejects_versions_and_unknown_fields(self):
        data = json.loads(dumps(gemm()))
        data["format"] = "prog-ir/99"
        with self.assertRaises(ValueError):
            loads(json.dumps(data))
        data = json.loads(dumps(gemm()))
        data["extra"] = 1
        with self.assertRaises(ValueError):
            loads(json.dumps(data))


if __name__ == "__main__":
    unittest.main()
