import unittest
from dataclasses import replace
from gtsim_frontend.ir.common import VerificationError
from gtsim_frontend.ir.execution import ExecProgram
from gtsim_frontend.programs.gemm_single import build
from gtsim_frontend.passes.instantiate import instantiate
from gtsim_frontend.passes.lower_hopper_gemm import lower
from gtsim_frontend.passes.lower_execution import lower as expand
from gtsim_frontend.serialization.json_codec import dumps, loads


class StageTests(unittest.TestCase):
    def test_all_stage_round_trips(self):
        program = build(128, 256, 128)
        flow = instantiate(program.launches[0].algorithm)
        native = lower(flow)
        for obj in (
            program,
            program.launches[0].algorithm,
            flow,
            native,
            expand(native),
            ExecProgram(symbol="empty"),
        ):
            self.assertEqual(obj, loads(dumps(obj)))

    def test_native_protocol_and_mapping_fail_closed(self):
        flow = instantiate(build().launches[0].algorithm)
        native = lower(flow)
        with self.assertRaises(VerificationError):
            replace(
                native, registers=replace(native.registers, fragment_shape=(1, 1))
            ).verify()
        with self.assertRaises(VerificationError):
            replace(
                native, async_protocol=replace(native.async_protocol, wait_group=1)
            ).verify()
        with self.assertRaises(VerificationError):
            replace(flow, mapping=replace(flow.mapping, execute=())).verify()
        with self.assertRaises(VerificationError):
            replace(
                native, resources=replace(native.resources, shared_bytes=1)
            ).verify()


if __name__ == "__main__":
    unittest.main()
