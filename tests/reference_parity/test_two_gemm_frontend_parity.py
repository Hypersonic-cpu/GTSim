#!/usr/bin/env python3
"""Compare the .progir compiler DAG with the legacy GEMM frontend."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "frontend/src"))

from parser import parse_file  # noqa: E402
from compile import compile_program as compile_new_program  # noqa: E402
from compile import lower_arch, lower_flow  # noqa: E402
from gtsim_frontend.driver import compile_program as compile_old_program  # noqa: E402
from gtsim_frontend.ir.algorithm import Algorithm  # noqa: E402
from gtsim_frontend.ir.common import Tensor  # noqa: E402
from gtsim_frontend.ir.program import Command, Invocation, Program  # noqa: E402


PROGRAM = ROOT / "frontend/asm/gemm/reference/two_gemms.progir"
ALGORITHM = ROOT / "frontend/asm/gemm/reference/gemm.algoir"
SCHEDULE = ROOT / "frontend/asm/gemm/reference/gemm.flowir"
DAG = ROOT / "frontend/asm/gemm/reference/two_gemms.dag.json"
RESOURCE_MAP = {
    "load": "LD",
    "store": "ST",
    "matrix": "Tensor",
    "arithmetic": "SIMD",
    "special_function": "SFU",
    "shuffle": "SHFL",
}


def old_program_from(prog):
    tensors = tuple(
        Tensor(
            tensor.id,
            tuple(tensor.shape),
            tensor.dtype.value,
            tensor.tier.value if tensor.tier else "HBM",
            tensor.read_only,
        )
        for tensor in prog.tensors
    )
    tensors_by_id = {tensor.id: tensor for tensor in prog.tensors}
    launches = []
    for invocation in prog.launches:
        attributes = dict(invocation.attributes)
        algorithm = Algorithm(
            symbol=invocation.id,
            dimensions=tuple(int(attributes[name]) for name in ("M", "N", "K")),
            dtype=tensors_by_id[invocation.inputs[0]].dtype.value,
        )
        launches.append(
            Invocation(
                invocation.id,
                algorithm,
                tuple(zip(("A", "B"), invocation.inputs))
                + (("C", invocation.outputs[0]),),
            )
        )
    command_names = {
        "launch": "LaunchKernel",
        "synchronize": "StreamSynchronize",
        "record_event": "RecordEvent",
        "wait_event": "WaitEvent",
    }
    commands = tuple(
        Command(
            command_names[command.kind.value],
            "" if command.kind.value == "synchronize" else command.target,
        )
        for command in prog.commands
    )
    result = Program(
        symbol=prog.symbol,
        streams=prog.streams,
        tensors=tensors,
        launches=tuple(launches),
        commands=commands,
    )
    result.verify()
    return result


def compare_kernel(test, new, region, old_path, bindings):
    old = json.loads(old_path.read_text())
    old_tensors = {tensor["id"]: tensor for tensor in old["tensors"]}
    new_tensors = {tensor.name: tensor for tensor in new.tensors}
    for tensor in old_tensors.values():
        actual = new_tensors[bindings[tensor["name"]]]
        test.assertEqual(
            (actual.shape, actual.strides, actual.dtype.value),
            (tuple(tensor["shape"]), tuple(tensor["strides"]), tensor["dtype"]),
        )
    new_ops = [new.ops[node] for node in region.nodes]
    begin = region.nodes[0]
    region_nodes = set(region.nodes)
    cta_map = {global_id: local_id for local_id, global_id in enumerate(region.ctas)}
    old_ctas = {cta["id"]: cta for cta in old["ctas"]}
    new_ctas = {cta.id: cta for cta in new.ctas}

    test.assertEqual(len(cta_map), len(old_ctas))
    for global_id, local_id in cta_map.items():
        actual, expected = new_ctas[global_id], old_ctas[local_id]
        test.assertEqual(
            (actual.sm, actual.warps, actual.shared_bytes, actual.register_bytes),
            (
                expected["sm"],
                expected["warps"],
                expected["shared_bytes"],
                expected["register_bytes"],
            ),
        )

    new_warp_base = {
        cta: min(op.warp for op in new_ops if op.cta == cta) for cta in cta_map
    }
    old_warp_base = {
        cta: min(op["warp"] for op in old["ops"] if op["cta"] == cta)
        for cta in cta_map.values()
    }
    test.assertEqual(len(new_ops), len(old["ops"]))
    for index, (actual, expected) in enumerate(zip(new_ops, old["ops"])):
        actual_tensor = (
            None if actual.tensor is None else new.tensors[actual.tensor].name
        )
        expected_tensor = (
            None
            if expected["tensor"] is None
            else bindings[old_tensors[expected["tensor"]]["name"]]
        )
        local_cta = cta_map[actual.cta]
        actual_op = (
            actual.opcode,
            RESOURCE_MAP[actual.resource.value],
            local_cta,
            actual.warp - new_warp_base[actual.cta] + old_warp_base[local_cta],
            actual_tensor,
            actual.dtype.value,
            actual.leading_dim or 0,
            tuple(actual.tile),
            frozenset(dep - begin for dep in actual.after_issue if dep in region_nodes),
            frozenset(
                dep - begin for dep in actual.after_complete if dep in region_nodes
            ),
        )
        expected_op = (
            expected["op"],
            expected["pipeline"],
            expected["cta"],
            expected["warp"],
            expected_tensor,
            expected["dtype"],
            expected["leading_dim"],
            tuple(expected["tile"]),
            frozenset(expected["after_issue"]),
            frozenset(expected["after_complete"]),
        )
        test.assertEqual(actual_op, expected_op, f"{region.id}: node {index}")

    old_outputs = tuple(
        bindings[old_tensors[tensor]["name"]] for tensor in old["outputs"]
    )
    new_outputs = tuple(new.tensors[tensor].name for tensor in region.outputs)
    test.assertEqual(new_outputs, old_outputs)


class TwoGemmFrontendParityTests(unittest.TestCase):
    def test_progir_lowering_matches_legacy_gemm_dags(self):
        prog = parse_file(PROGRAM)
        new_flow = compile_new_program(PROGRAM, {"NUM_SMS": 1}, [ALGORITHM], [SCHEDULE])
        new = lower_arch(lower_flow(new_flow))
        self.assertEqual(parse_file(DAG), new)
        old = old_program_from(prog)

        # The legacy pass checks only resource limits here; no timing is simulated.
        profile = {
            "num_sms": 1,
            "shared_memory_per_cta_bytes": 232448,
            "register_file_per_sm_bytes": 262144,
            "max_warps_per_sm": 64,
        }
        with tempfile.TemporaryDirectory() as directory:
            old_execution, _ = compile_old_program(old, directory, profile)
            self.assertEqual(
                [launch.id for launch in old_execution.launches],
                [region.id for region in new.regions],
            )
            bindings = tuple(
                dict(zip(("A", "B", "C"), (*invocation.inputs, *invocation.outputs)))
                for invocation in prog.launches
            )
            for region, launch, names in zip(new.regions, old.launches, bindings):
                compare_kernel(
                    self,
                    new,
                    region,
                    Path(directory) / launch.id / "kernel.gtsir",
                    names,
                )

            self.assertEqual(new.regions[0].after, ())
            self.assertEqual(new.regions[1].after, (new.regions[0].id,))
            self.assertEqual(
                dict(old_execution.launches[0].bindings)[2],
                dict(old_execution.launches[1].bindings)[0],
            )
            cross_region = {
                dependency
                for node_id in new.regions[1].nodes
                for dependency in (
                    *new.ops[node_id].after_issue,
                    *new.ops[node_id].after_complete,
                )
                if dependency not in new.regions[1].nodes
            }
            self.assertTrue(cross_region)
            self.assertTrue(
                all(
                    new.ops[node].opcode == "cp.sram2dram_tma"
                    and new.tensors[new.ops[node].tensor].name == "H"
                    for node in cross_region
                )
            )


if __name__ == "__main__":
    unittest.main()
