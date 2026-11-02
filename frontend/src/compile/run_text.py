"""Compile a textual Program and expand its kernels in RAM, without DAG files."""

import argparse
import json
from pathlib import Path
from parser import serialize_file
from . import compile_program, lower_flow
from .arch_to_exec import lower_region


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--lib", type=Path, required=True)
    parser.add_argument("--params", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--emit-ir", action="store_true", help="Write only compact FlowIR/ArchIR"
    )
    args = parser.parse_args()
    flow = compile_program(
        args.source,
        json.loads(args.params.read_text()),
        list(args.lib.glob("*.algoir")),
        list(args.lib.glob("*.flowir")),
    )
    arch = lower_flow(flow)
    if args.emit_ir:
        if args.out is None:
            parser.error("--emit-ir requires --out")
        args.out.mkdir(parents=True, exist_ok=True)
        serialize_file(flow, args.out / "block.flowir")
        serialize_file(arch, args.out / "block.nvir")
    count = 0
    for region in arch.regions:
        execution = lower_region(arch, region)
        if execution.ops:
            print(
                json.dumps(
                    dict(id=region.symbol, nodes=len(execution.ops), after=region.after)
                ),
                flush=True,
            )
            count += len(execution.ops)
        del execution
    print(json.dumps(dict(symbol=arch.symbol, total_nodes=count)))


if __name__ == "__main__":
    main()
