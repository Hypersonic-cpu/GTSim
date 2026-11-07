"""Independent typed AST parsing for the checked-in assembly examples."""

from dataclasses import replace, fields, is_dataclass, MISSING
from pathlib import Path
import subprocess
import json
import sys
from tempfile import TemporaryDirectory
import unittest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
import parser
from parser import ast as A
from parser.json_codec import dumps

REFERENCE = SRC.parent / "asm/gemm/reference"


class ASTParserTests(unittest.TestCase):
    def test_three_assembly_sources_create_typed_semantic_objects(self):
        prog = parser.parse_file(REFERENCE / "example.progir")
        self.assertIs(type(prog), A.ProgIR)
        self.assertEqual(
            tuple((p.id, p.dtype) for p in prog.static.parameters),
            (("M", "index"), ("N", "index"), ("K", "index")),
        )
        self.assertIs(type(prog.tensors[0]), A.Tensor)
        self.assertEqual(prog.tensors[0].shape, ("M", "K"))
        self.assertEqual(prog.launches[0].callee, "gemm_reference")
        self.assertEqual(prog.launches[0].implementation, "gemm_reference_flow")
        self.assertEqual(prog.commands[1].after, ("e0",))
        algo = parser.parse_file(REFERENCE / "gemm.algoir")
        self.assertIs(type(algo), A.AlgoIR)
        self.assertEqual(tuple(d[2] for d in algo.domains), ("M", "N", "K"))
        self.assertEqual(len(algo.computations), 2)
        self.assertIn("sum_k(", algo.computations[0].expression)
        self.assertEqual(algo.stages[1].after, ("operand_acquisition",))
        flow = parser.parse_file(REFERENCE / "gemm.flowir")
        self.assertIs(type(flow), A.FlowIR)
        self.assertEqual(len(flow.regions[0].ops), 6)
        self.assertEqual(flow.regions[0].config.tile_shape, (128, 256, 64))
        self.assertEqual(flow.regions[0].domain.pipeline.depth, 2)
        self.assertEqual(flow.regions[0].ops[1].region.shape, (128, 64))
        self.assertIs(type(flow.regions[0].ops[1].dependencies[0]), A.Dependency)
        self.assertEqual(flow.regions[0].ops[1].dependencies[0].source, "matmul")
        self.assertEqual(flow.regions[0].ops[3].operands[-1].iteration_delta, -1)
        self.assertEqual(
            dict(flow.regions[0].mapping.execute)[A.ExecutorRole.MATRIX], "WGMMA"
        )

    def test_assembly_roundtrip_and_mutations_use_fields(self):
        for file in ("example.progir", "gemm.algoir", "gemm.flowir"):
            obj = parser.parse_file(REFERENCE / file)
            self.assertEqual(parser.parse(parser.serialize(obj)), obj)
            self.assertEqual(parser.parse(dumps(obj)), obj)
            changed = replace(obj, symbol=obj.symbol + "_edited")
            if isinstance(changed, A.FlowIR):
                changed = replace(
                    changed,
                    regions=(
                        replace(changed.regions[0], mapping_symbol="edited_mapping"),
                    ),
                )
            self.assertEqual(parser.parse(parser.serialize(changed)), changed)
        flow = parser.parse_file(REFERENCE / "gemm.flowir")
        changed = replace(
            flow,
            regions=(
                replace(
                    flow.regions[0],
                    config=replace(flow.regions[0].config, tile_shape=(64, 256, 64)),
                ),
            ),
        )
        self.assertEqual(
            parser.parse(parser.serialize(changed)).regions[0].config.tile_shape,
            (64, 256, 64),
        )
        with TemporaryDirectory() as temp:
            output = Path(temp) / "copy.flowir"
            parser.serialize_file(changed, output)
            self.assertEqual(parser.parse_file(output), changed)

    def test_static_conditions_are_typed_and_roundtrip_from_fields(self):
        source = (REFERENCE / "gemm.flowir").read_text()
        flow = parser.parse(source)
        self.assertIs(type(flow.static.predicates[0]), A.StaticPredicate)
        self.assertIs(type(flow.regions[0].static_ifs[0]), A.StaticIf)
        self.assertEqual(flow.regions[0].static_ifs[0].condition, "%valid_tile")
        self.assertEqual(
            flow.regions[0].dependency_rules[0].static_if, "%has_previous_k"
        )
        self.assertIn(
            A.StaticPredicate(
                "odd_k_count",
                "%k_count mod 2 == 1",
            ),
            flow.static.predicates,
        )
        changed = replace(
            flow,
            static=replace(
                flow.static,
                predicates=(
                    replace(flow.static.predicates[0], expression="%tile < 7"),
                    *flow.static.predicates[1:],
                ),
            ),
            regions=(
                replace(
                    flow.regions[0],
                    dependency_rules=(
                        replace(flow.regions[0].dependency_rules[0], static_if="false"),
                        *flow.regions[0].dependency_rules[1:],
                    ),
                ),
            ),
        )
        text = parser.serialize(changed)
        self.assertIn("static.pred %valid_tile = %tile < 7", text)
        self.assertIn("static.if <false>", text)
        self.assertNotIn("when<", text)
        self.assertEqual(parser.parse(text), changed)
        self.assertEqual(parser.parse(dumps(changed)), changed)
        with self.assertRaises(parser.ParseError):
            parser.parse(
                source.replace("static.if <%has_previous_k>", "when <%has_previous_k>")
            )

    def test_shared_static_branch_syntax_and_flow_alternatives(self):
        from parser import syntax

        for dialect in ("prog", "algo", "flow", "arch", "exec"):
            text = f"""{dialect}.kernel @example {{
  static.if <%B == 1> {{
    {dialect}.operation @one
  }}
  static.elif <%B == 8> {{
    {dialect}.operation @eight
  }}
  static.else {{
    {dialect}.operation @other
  }}
}}
"""
            module = syntax.parse(text)
            branches = module.nodes[0].body
            self.assertEqual(
                tuple(branch.kind for branch in branches),
                (
                    A.StaticBranchKind.IF,
                    A.StaticBranchKind.ELIF,
                    A.StaticBranchKind.ELSE,
                ),
            )
            self.assertEqual(syntax.parse(syntax.serialize(module)), module)
        with self.assertRaises(ValueError):
            syntax.parse("static.else {\n}\n")
        with self.assertRaises(ValueError):
            syntax.parse(
                "static.if <true> {\n}\nstatic.else {\n}\nstatic.elif <false> {\n}\n"
            )

        source = (REFERENCE / "gemm.flowir").read_text()
        first = """  static.if <%has_previous_k> {
    flow.dependency_rule @previous_load from[@load_A.complete, @load_B.complete] at[%wave = %wave, %k = %k-1] select<all>
  }"""
        alternatives = (
            first
            + """
  static.elif <%previous_wave_start> {
    flow.dependency_rule @alternative from[@load_A.complete] at[%k = 0] select<all>
  }
  static.else {
    flow.dependency_rule @fallback from[@load_B.complete] at[%k = 0] select<all>
  }"""
        )
        flow = parser.parse(source.replace(first, alternatives))
        self.assertEqual(flow.regions[0].dependency_rules[1].id, "alternative")
        self.assertIn("not", flow.regions[0].dependency_rules[1].static_if)
        self.assertIn("%has_previous_k", flow.regions[0].dependency_rules[2].static_if)
        self.assertIn(
            "%previous_wave_start", flow.regions[0].dependency_rules[2].static_if
        )
        self.assertEqual(parser.parse(parser.serialize(flow)), flow)

    def test_static_declarations_use_one_parser_and_one_shared_container(self):
        declaration = """  static.param %compiler_input : index range[1, 16]
  static.let %compiler_count = %compiler_input * 2
  static.pred %compiler_enabled = %compiler_count > 1
  static.require <%compiler_count > 0>
"""
        for filename in ("example.progir", "gemm.algoir", "gemm.flowir"):
            source = (REFERENCE / filename).read_text()
            source = source.replace("{\n", "{\n" + declaration, 1)
            root = parser.parse(source)
            self.assertIs(type(root.static), A.StaticDeclarations)
            self.assertEqual(
                root.static.parameters[0],
                A.StaticParameter("compiler_input", A.DataType.INDEX, "1", "16"),
            )
            self.assertIn(
                A.StaticBinding("compiler_count", "%compiler_input * 2"),
                root.static.bindings,
            )
            self.assertIn(
                A.StaticPredicate("compiler_enabled", "%compiler_count > 1"),
                root.static.predicates,
            )
            self.assertIn("%compiler_count > 0", root.static.constraints)
            self.assertEqual(parser.parse(parser.serialize(root)), root)
        for cls in (A.ProgIR, A.AlgoIR, A.FlowIR, A.ArchIR, A.ExecIR):
            self.assertEqual(
                next(f for f in fields(cls) if f.name == "static").type,
                A.StaticDeclarations,
            )

    def test_all_five_ir_class_structural_roundtrip(self):
        prog = parser.parse_file(REFERENCE / "example.progir")
        algo = parser.parse_file(REFERENCE / "gemm.algoir")
        flow = parser.parse_file(REFERENCE / "gemm.flowir")
        from compile import lower_flow

        arch = lower_flow(flow)
        execution = A.ExecIR(
            "dag",
            A.StaticDeclarations((), (), (), (), ()),
            "sm90",
            (),
            (),
            (),
            (),
            (),
            (),
            None,
            (),
        )
        for obj in (prog, algo, flow, arch, execution):
            parsed = parser.parse(dumps(obj))
            self.assertIs(type(parsed), type(obj))
            self.assertEqual(parsed, obj)
        for obj in (arch, execution):
            self.assertEqual(parser.parse(parser.serialize(obj)), obj)

    def test_ast_has_no_implicit_defaults_or_compatibility_fields(self):
        for name, cls in vars(A).items():
            if isinstance(cls, type) and is_dataclass(cls):
                for field in fields(cls):
                    with self.subTest(record=name, field=field.name):
                        self.assertIs(field.default, MISSING)
                        self.assertIs(field.default_factory, MISSING)
                        self.assertNotIn(
                            field.name,
                            (
                                "schema_version",
                                "source_hash",
                                "family",
                                "sync_semantics_version",
                            ),
                        )
        for name in (
            "GEMM",
            "FA3",
            "ReferenceProfile",
            "AttentionPartition",
            "RopeSpec",
            "OpRole",
        ):
            self.assertFalse(hasattr(A, name), name)

    def test_arithmetic_special_functions_and_constants(self):
        algo = parser.parse_file(REFERENCE / "gemm.algoir")
        operations = (
            A.MathOperation(
                "integer_add_sub_shift",
                A.OperationClass.ARITHMETIC,
                A.ArithmeticCost.SIMPLE,
                A.DataType.INT32,
                ("x", "y"),
                "z",
                (),
            ),
            A.MathOperation(
                "float_add_sub",
                A.OperationClass.ARITHMETIC,
                A.ArithmeticCost.SIMPLE,
                A.DataType.FP32,
                ("x", "y"),
                "z",
                (),
            ),
            A.MathOperation(
                "divide",
                A.OperationClass.ARITHMETIC,
                A.ArithmeticCost.DIVIDE,
                A.DataType.FP32,
                ("x", "y"),
                "z",
                (),
            ),
            A.MathOperation(
                "sin_cos",
                A.OperationClass.SPECIAL_FUNCTION,
                A.SpecialFunction.TRIGONOMETRIC,
                A.DataType.FP32,
                ("x",),
                "z",
                (),
            ),
            A.MathOperation(
                "exp",
                A.OperationClass.SPECIAL_FUNCTION,
                A.SpecialFunction.EXPONENTIAL,
                A.DataType.FP32,
                ("x",),
                "z",
                (),
            ),
            A.MathOperation(
                "log",
                A.OperationClass.SPECIAL_FUNCTION,
                A.SpecialFunction.LOGARITHMIC,
                A.DataType.FP32,
                ("x",),
                "z",
                (),
            ),
        )
        changed = replace(algo, operations=operations)
        self.assertEqual(parser.parse(parser.serialize(changed)), changed)
        self.assertEqual(parser.parse(dumps(changed)), changed)
        self.assertNotIn("SIN", A.OperationClass.__members__)
        changed = replace(
            algo,
            accumulators=(
                A.Constant("acc", A.ConstantValue.NEGATIVE_INFINITY, A.DataType.FP32),
            ),
        )
        self.assertEqual(parser.parse(parser.serialize(changed)), changed)

    def test_bad_input_has_parser_errors(self):
        for text in (
            "",
            "}",
            "prog.program @x params() {",
            "unknown.kernel @x {}",
            '{"format":"bad"}',
        ):
            with self.subTest(text=text), self.assertRaises(parser.ParseError):
                parser.parse(text)
        original = (REFERENCE / "gemm.flowir").read_text()
        with self.assertRaises(parser.ParseError):
            parser.parse(
                original.replace(
                    "@matmul.complete:buffer_reuse", "@matmul.missing:buffer_reuse"
                )
            )
        with self.assertRaisesRegex(parser.ParseError, "missing.flowir"):
            parser.parse_file(REFERENCE / "missing.flowir")

    def test_flow_names_and_tile_geometry_are_not_workload_presets(self):
        text = (REFERENCE / "gemm.flowir").read_text()
        text = (
            text.replace("load_A", "read_left")
            .replace("load_B", "read_right")
            .replace("ab_stage", "staging")
            .replace("%k", "%reduction")
        )
        text = text.replace("tile(128x256x64)", "tile(64x32x16)")
        flow = parser.parse(text)
        self.assertEqual(flow.regions[0].config.tile_shape, (64, 32, 16))
        self.assertEqual(flow.regions[0].ops[1].id, "read_left")
        self.assertEqual(flow.regions[0].ops[1].slot.buffer, "staging")
        self.assertEqual(flow.regions[0].domain.pipeline.loop.variable, "reduction")
        self.assertEqual(parser.parse(parser.serialize(flow)), flow)

    def test_raw_strings_legacy_fields_and_missing_config_are_rejected(self):
        obj = parser.parse_file(REFERENCE / "gemm.flowir")
        raw = json.loads(dumps(obj))
        raw["tensors"][0]["dtype"] = "fp16"
        with self.assertRaisesRegex(parser.ParseError, "DataType"):
            parser.parse(json.dumps(raw))
        raw = json.loads(dumps(obj))
        raw["schema_version"] = 2
        with self.assertRaisesRegex(parser.ParseError, "unknown fields"):
            parser.parse(json.dumps(raw))
        raw = json.loads(dumps(obj))
        raw["family"] = "gemm"
        with self.assertRaisesRegex(parser.ParseError, "unknown fields"):
            parser.parse(json.dumps(raw))
        with self.assertRaisesRegex(parser.ParseError, "schedule_config"):
            parser.parse(
                (REFERENCE / "gemm.flowir").read_text().replace("stages(2)", "")
            )
        algo = parser.parse_file(REFERENCE / "gemm.algoir")
        bad = A.MathOperation(
            "bad",
            A.OperationClass.SPECIAL_FUNCTION,
            A.ArithmeticCost.SIMPLE,
            A.DataType.FP32,
            ("x",),
            "y",
            (),
        )
        with self.assertRaisesRegex(parser.ParseError, "incompatible"):
            parser.parse(dumps(replace(algo, operations=(bad,))))

    def test_no_legacy_imports_and_no_stdlib_ast_shadowing(self):
        code = """
import sys, importlib.abc
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.startswith("gtsim_frontend"):raise AssertionError(fullname)
sys.meta_path.insert(0,Block())
import parser,ast,dataclasses
assert hasattr(ast,"parse")
for file in ("example.progir","gemm.algoir","gemm.flowir"):
 obj=parser.parse_file(sys.argv[1]+"/"+file)
 assert parser.parse(parser.serialize(obj))==obj
assert not any(n.startswith("gtsim_frontend") for n in sys.modules)
"""
        subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys;sys.path.insert(0," + repr(str(SRC)) + ");" + code,
                str(REFERENCE),
            ],
            check=True,
        )


if __name__ == "__main__":
    unittest.main()
