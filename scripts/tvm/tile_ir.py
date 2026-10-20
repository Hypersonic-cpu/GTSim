"""File-only writer for the version 1 contract (no simulator/TVM imports)."""

from dataclasses import dataclass, field
from pathlib import Path


def vector(values):
    return "[" + ",".join(str(v) for v in values) + "]"


@dataclass
class Operation:
    name: str
    op: str
    cta: int
    warp: int
    tensor: int
    tile: tuple
    dtype: str = "fp16"
    after_complete: list = field(default_factory=list)
    after_issue: list = field(default_factory=list)


class TileIR:
    def __init__(self):
        self.tensors = []
        self.ctas = []
        self.operations = []

    def tensor(self, ident, name, shape, strides, dtype="fp16"):
        self.tensors.append((ident, name, shape, strides, dtype))

    def cta(self, ident, sm, shared, registers):
        self.ctas.append((ident, sm, shared, registers))

    def add(self, operation):
        ident = len(self.operations)
        self.operations.append(operation)
        return ident

    def text(self):
        lines = ["gtsim.tile_ir 1"]
        for ident, name, shape, strides, dtype in self.tensors:
            lines.append(f"tensor id={ident} name={name} shape={vector(shape)} "
                         f"strides={vector(strides)} dtype={dtype} tier=HBM")
        for ident, sm, shared, registers in self.ctas:
            lines.append(f"cta id={ident} sm={sm} warps=12 shared={shared} registers={registers}")
        for ident, op in enumerate(self.operations):
            lines.append(f"%{ident} = {op.op} name={op.name} cta={op.cta} warp={op.warp} "
                         f"tensor={op.tensor} dtype={op.dtype} tile={vector(op.tile)} "
                         f"after_issue={vector(op.after_issue)} after_complete={vector(op.after_complete)}")
        return "\n".join(lines) + "\n"

    def write(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.text())
