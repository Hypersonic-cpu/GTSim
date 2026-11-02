"""Compile a textual program and explicit algorithm/schedule libraries to FlowIR."""

import argparse
import json
from parser import serialize_file
from . import compile_program


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("program")
    cli.add_argument("--algo", action="append", required=True)
    cli.add_argument("--flow", action="append", required=True)
    cli.add_argument("--param", action="append", default=[], metavar="NAME=JSON_VALUE")
    cli.add_argument("-o", "--output", required=True)
    args = cli.parse_args()
    parameters = {
        name: json.loads(value)
        for name, value in (entry.split("=", 1) for entry in args.param)
    }
    flow = compile_program(args.program, parameters, args.algo, args.flow)
    serialize_file(flow, args.output)
    print(
        f"{args.output}: {len(flow.regions)} regions, {sum(len(r.ops) for r in flow.regions)} structured operations"
    )


if __name__ == "__main__":
    main()
