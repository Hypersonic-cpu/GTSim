"""Independent checks of schedule instantiation, branching and composition."""

from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
from compile import compile_program
from compile.static import Resolver, EMPTY, specialize_source
from parser import ast as A, parse_file, parse, serialize, from_module
from parser.json_codec import dumps

REFERENCE = SRC.parent / "asm/gemm/reference"


def build(program=None, sms=1):
    return compile_program(
        program or REFERENCE / "two_gemms.progir",
        {"NUM_SMS": sms},
        [REFERENCE / "gemm.algoir"],
        [REFERENCE / "gemm.flowir"],
    )


class ProgramToFlowTests(unittest.TestCase):
    def test_two_gemms_reuse_one_schedule_and_share_actual_intermediate(self):
        flow = build()
        self.assertIs(type(flow), A.FlowIR)
        self.assertEqual(flow.stream, "compute")
        self.assertEqual(flow.outputs, ("Y",))
        first, second = flow.regions
        self.assertEqual(first.after, ())
        self.assertEqual(second.after, ("first",))
        self.assertEqual((first.origin, second.origin), ("gemm_reference",) * 2)
        self.assertEqual(tuple(len(r.ops) for r in flow.regions), (6, 6))
        self.assertEqual(
            (first.domain.pipeline.loop.stop, second.domain.pipeline.loop.stop), (3, 4)
        )
        self.assertEqual(first.domain.pipeline.stage.generation_counts_per_wave, (2, 1))
        self.assertEqual(
            second.domain.pipeline.stage.generation_counts_per_wave, (2, 2)
        )
        self.assertEqual(first.ops[-1].region.tensor, "H")
        self.assertEqual(second.ops[1].region.tensor, "H")
        self.assertEqual(first.ops[3].math.inputs, ("X", "W0"))
        self.assertEqual(second.ops[3].math.inputs, ("H", "W1"))
        self.assertEqual(
            (first.ops[-1].math.output, second.ops[-1].math.output), ("H", "Y")
        )
        self.assertEqual(
            [t.shape for t in flow.tensors],
            [(256, 192), (192, 256), (256, 256), (256, 512), (256, 512)],
        )
        self.assertEqual(parse(serialize(flow)), flow)
        self.assertEqual(parse(dumps(flow)), flow)
        self.assertNotIn("%K", serialize(flow))
        self.assertNotIn("prog.launch", serialize(flow))

    def test_parameter_branch_selects_different_buffer_dependencies(self):
        first, second = build().regions
        self.assertEqual(
            len([r for r in first.dependency_rules if r.id == "buffer_reuse"]), 2
        )
        self.assertEqual(
            len([r for r in second.dependency_rules if r.id == "buffer_reuse"]), 3
        )
        # Independent oracle for the selected cross-wave ping-pong edges.
        expected = ({0: 2}, {0: 2, 1: 3})
        for region, oracle in zip((first, second), expected):
            for k in (0, 1):
                resolver = Resolver(region.static, {"k": k, "wave": 1})
                edges = [
                    dict(rule.producer_bindings)
                    for rule in region.dependency_rules
                    if rule.id == "buffer_reuse" and resolver.value(rule.static_if)
                ]
                self.assertEqual(
                    [resolver.value(edge["k"]) for edge in edges],
                    [oracle[k]] if k in oracle else [],
                )

    def test_program_static_if_elif_else_uses_the_same_macro_resolver(self):
        original = (REFERENCE / "two_gemms.progir").read_text()
        first = original.index("    %H, !first_done")
        second = original.index("    %Y, !second_done")
        launch = original[first:second]
        variant = (
            original[:first]
            + "    static.if <%NUM_SMS == 1> {\n"
            + launch
            + "    }\n    static.elif <%NUM_SMS == 2> {\n"
            + launch.replace("N=256", "N=512")
            + "    }\n    static.else {\n"
            + launch.replace("N=256", "N=768")
            + "    }\n"
            + original[second:]
        )
        with TemporaryDirectory() as temp:
            path = Path(temp) / "branch.progir"
            path.write_text(variant)
            for sms, expected in ((1, "256"), (2, "512"), (3, "768")):
                module, _ = specialize_source(path, {"NUM_SMS": sms})
                program = from_module(module)
                self.assertEqual(dict(program.launches[0].attributes)["N"], expected)
            self.assertEqual(len(build(path).regions), 2)

    def test_generic_expression_folding_and_constraints(self):
        declarations = A.StaticDeclarations(
            (A.StaticParameter("K", A.DataType.INDEX, "64", "*"),),
            (A.StaticBinding("count", "ceildiv(%K,64)"),),
            (A.StaticPredicate("odd", "%count mod 2 == 1"),),
            ("%K divisible_by 64",),
        )
        self.assertIs(Resolver(declarations, {"K": 192}).value("%odd"), True)
        self.assertIs(Resolver(declarations, {"K": 256}).value("%odd"), False)
        self.assertIs(Resolver(EMPTY, {}).value("false and missing_function()"), False)
        with self.assertRaises(ValueError):
            Resolver(declarations, {"K": 193})
        with self.assertRaises(ValueError):
            Resolver(EMPTY, {}).value('__import__("os")')

    def test_shape_mismatch_is_rejected_before_composition(self):
        with TemporaryDirectory() as temp:
            path = Path(temp) / "wrong.progir"
            path.write_text(
                (REFERENCE / "two_gemms.progir")
                .read_text()
                .replace("tensor<256x192xfp16>", "tensor<256x128xfp16>")
            )
            with self.assertRaisesRegex(ValueError, "tensor binding mismatch"):
                build(path)

    def test_target_parameter_changes_cta_allocation_without_changing_shapes(self):
        one, two = build(sms=1), build(sms=2)
        self.assertEqual(one.tensors, two.tensors)
        self.assertEqual([r.domain.loops[0].stop for r in one.regions], [1, 1])
        self.assertEqual([r.domain.loops[0].stop for r in two.regions], [2, 2])


if __name__ == "__main__":
    unittest.main()
