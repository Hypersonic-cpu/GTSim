"""Compile Program/Flow/Arch IR using explicit stages; expand only for ExecIR."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
from parser import ast as A, parse_file, serialize_file
from . import compile_program, lower_flow, lower_arch
from .export_runtime import export_runtime


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("source")
    cli.add_argument("--params", type=Path)
    cli.add_argument("--lib", type=Path)
    cli.add_argument("--algo", action="append", default=[])
    cli.add_argument("--flow", action="append", default=[])
    cli.add_argument("--param", action="append", default=[], metavar="NAME=JSON_VALUE")
    cli.add_argument("--runtime-dir", type=Path)
    cli.add_argument("--prefetch-config", type=Path)
    cli.add_argument("--launch-overhead", type=int, default=0)
    cli.add_argument("-o", "--output", required=True)
    args = cli.parse_args()
    parameters = {
        name: json.loads(value) for name, value in (v.split("=", 1) for v in args.param)
    }
    if args.params:
        parameters = {**json.loads(args.params.read_text()), **parameters}
    if args.lib:
        args.algo += [str(p) for p in args.lib.glob("*.algoir")]
        args.flow += [str(p) for p in args.lib.glob("*.flowir")]
    source = parse_file(args.source)
    if isinstance(source, A.ProgIR):
        source = compile_program(args.source, parameters, args.algo, args.flow)
    elif parameters:
        bound = {v.id: v.expression for v in source.static.arguments}
        bound.update((name, repr(value)) for name, value in parameters.items())
        source = replace(
            source,
            static=replace(
                source.static,
                arguments=tuple(
                    A.StaticBinding(name, value) for name, value in bound.items()
                ),
            ),
        )
    suffix = Path(args.output).suffix
    prefetch = None
    if args.prefetch_config:
        if (
            not isinstance(source, A.FlowIR)
            or not args.runtime_dir
            or suffix in (".flowir", ".nvir")
        ):
            cli.error(
                "--prefetch-config requires Program/Flow input and runtime export"
            )
        from .opt.prefetch import Residency, optimize

        prefetch = optimize(
            source, Residency(**json.loads(args.prefetch_config.read_text()))
        )
        source = prefetch.flow
    if isinstance(source, A.FlowIR) and suffix != ".flowir":
        source = lower_flow(source)
    if isinstance(source, A.ArchIR) and suffix != ".nvir":
        source = lower_arch(source)
    serialize_file(source, args.output)
    if args.runtime_dir:
        if not isinstance(source, A.ExecIR):
            cli.error("--runtime-dir requires ExecIR output")
        export_runtime(source, args.runtime_dir, args.launch_overhead, prefetch)
    print(f"{args.output}: {type(source).__name__}")


if __name__ == "__main__":
    main()
