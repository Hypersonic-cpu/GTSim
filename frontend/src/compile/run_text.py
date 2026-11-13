"""Compile textual IR and count native instructions through streaming callbacks."""

import argparse
import json
from dataclasses import replace
from pathlib import Path
from parser import serialize_file
from . import compile_program, lower_flow
from .expand_native import NativeExpander
from .static import EMPTY, Resolver, bound_resolver
from .streaming import _strides
from parser import ast as A


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--lib", type=Path, required=True)
    parser.add_argument("--params", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--emit-ir", action="store_true")
    args = parser.parse_args()
    flow = compile_program(args.source, json.loads(args.params.read_text()),
                           list(args.lib.glob("*.algoir")), list(args.lib.glob("*.flowir")))
    arch = lower_flow(flow)
    if args.emit_ir:
        if args.out is None:
            parser.error("--emit-ir requires --out")
        args.out.mkdir(parents=True, exist_ok=True)
        serialize_file(flow, args.out / "block.flowir")
        serialize_file(arch, args.out / "block.nvir")
    root = bound_resolver(arch.static, Resolver(EMPTY, {})).materialize_constants()
    tensors = {}
    for spec in arch.tensors:
        shape = tuple(root.value(d) for d in spec.shape)
        tensors[spec.id] = A.KernelTensor(
            spec.id, spec.id, shape, _strides(shape), spec.dtype,
            spec.tier or A.MemorySpace.HBM, spec.external, spec.read_only, None,
        )
    total = 0
    for region in arch.regions:
        arguments = {a.id: root.value(a.expression) for a in region.static.arguments}
        resolver = Resolver(replace(region.static, parameters=(), constraints=()),
                            {**root.values, **arguments}).materialize_constants()
        cta_ids = []
        expander = NativeExpander(region, resolver, tensors)
        expander.expand_to(lambda c: cta_ids.append(c.id), lambda _: None)
        node_count = expander.next_node_id
        cta_count = len(cta_ids)
        if node_count:
            print(json.dumps(dict(id=region.symbol, nodes=node_count, ctas=cta_count,
                                  after=region.after)), flush=True)
            total += node_count
    print(json.dumps(dict(symbol=arch.symbol, total_nodes=total)))
if __name__ == "__main__":
    main()
