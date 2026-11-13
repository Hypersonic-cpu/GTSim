"""Compile ProgramIR/FlowIR into the compact FlowIR or ArchIR stages."""

import argparse
import json
from dataclasses import replace
from pathlib import Path
from parser import ast as A, parse_file, serialize_file
from . import compile_program, lower_flow


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("source")
    cli.add_argument("--params", type=Path)
    cli.add_argument("--lib", type=Path)
    cli.add_argument("--algo", action="append", default=[])
    cli.add_argument("--flow", action="append", default=[])
    cli.add_argument("--param", action="append", default=[], metavar="NAME=JSON_VALUE")
    cli.add_argument("-o", "--output", required=True)
    args = cli.parse_args()
    parameters = {name: json.loads(value) for name, value in
                  (item.split("=", 1) for item in args.param)}
    if args.params:
        parameters = {**json.loads(args.params.read_text()), **parameters}
    if args.lib:
        args.algo += [str(p) for p in args.lib.glob("*.algoir")]
        args.flow += [str(p) for p in args.lib.glob("*.flowir")]
    source = parse_file(args.source)
    if isinstance(source, A.ProgIR):
        source = compile_program(args.source, parameters, args.algo, args.flow)
    elif parameters:
        bindings = {value.id: value.expression for value in source.static.arguments}
        bindings.update((name, repr(value)) for name, value in parameters.items())
        source = replace(source, static=replace(
            source.static,
            arguments=tuple(A.StaticBinding(name, value) for name, value in bindings.items()),
        ))
    suffix = Path(args.output).suffix
    if isinstance(source, A.FlowIR) and suffix != ".flowir":
        source = lower_flow(source)
    if isinstance(source, A.ArchIR) and suffix == ".flowir":
        cli.error("cannot write ArchIR as .flowir")
    serialize_file(source, args.output)
    print(f"{args.output}: {type(source).__name__}")


if __name__ == "__main__":
    main()
