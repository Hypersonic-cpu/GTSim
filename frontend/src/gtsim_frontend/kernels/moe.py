"""Canonical Router, assignment baseline and weighted expert-reduction schedules."""

from dataclasses import replace
from collections import Counter
from ..ir.primitives import RouterSpec, HistogramSpec, GatherSpec, WeightedSpec
from ..ir.common import Tensor, require
from ..ir.algorithm import (
    AlgorithmContract,
    OperandContract,
    IndexDomain,
    MathKind as M,
)
from ..ir.schedule import (
    WorkGroup,
    LoopVar as V,
    IndexExpr,
    Dependency,
    Phase,
    Relation,
)


def verify_spec(spec):
    require(spec.batch > 0 and spec.top_k > 0, "moe", "invalid token/assignment count")
    if isinstance(spec, (RouterSpec, HistogramSpec, GatherSpec)):
        require(spec.experts >= spec.top_k, "moe", "TopK exceeds expert count")
    if isinstance(spec, GatherSpec):
        require(
            spec.width > 0 and len(spec.assignments) == spec.batch * spec.top_k,
            "gather",
            "invalid assignment count",
        )
        require(
            sorted((a.token, a.slot) for a in spec.assignments)
            == [(b, k) for b in range(spec.batch) for k in range(spec.top_k)],
            "gather",
            "token/slot assignment identity mismatch",
        )
        require(
            all(0 <= a.expert < spec.experts for a in spec.assignments),
            "gather",
            "invalid expert",
        )
        for token in range(spec.batch):
            require(
                len({a.expert for a in spec.assignments if a.token == token})
                == spec.top_k,
                "gather",
                "duplicate token expert",
            )
    if isinstance(spec, WeightedSpec):
        require(
            spec.width > 0
            and len(spec.rows_by_token) == spec.batch
            and all(len(rows) == spec.top_k for rows in spec.rows_by_token),
            "weighted",
            "invalid assignment rows",
        )
        require(
            sorted(row for rows in spec.rows_by_token for row in rows)
            == list(range(spec.batch * spec.top_k)),
            "weighted",
            "duplicate/missing expert output row",
        )


def signature(spec):
    if isinstance(spec, RouterSpec):
        return (
            Tensor("Logits", (spec.batch, spec.experts), "bf16", read_only=True),
            Tensor("Probabilities", (spec.batch, spec.top_k), "bf16"),
            Tensor("ExpertIDs", (spec.batch, spec.top_k), "int32"),
        )
    if isinstance(spec, HistogramSpec):
        return (
            Tensor("ExpertIDs", (spec.batch, spec.top_k), "int32", read_only=True),
            Tensor("Counts", (1, spec.experts), "int32"),
            Tensor("Offsets", (1, spec.experts + 1), "int32"),
        )
    if isinstance(spec, GatherSpec):
        count = spec.batch * spec.top_k
        return (
            Tensor("X", (spec.batch, spec.width), "bf16", read_only=True),
            Tensor("ExpertIDs", (spec.batch, spec.top_k), "int32", read_only=True),
            Tensor("Probabilities", (spec.batch, spec.top_k), "bf16", read_only=True),
            Tensor("Offsets", (1, spec.experts + 1), "int32", read_only=True),
            Tensor("SortedX", (count, spec.width), "bf16"),
            Tensor("Assignments", (count, 3), "int32"),
            Tensor("SortedProbabilities", (count, 1), "bf16"),
        )
    ts = (
        Tensor(
            "ExpertOutput",
            (spec.batch * spec.top_k, spec.width),
            "bf16",
            read_only=True,
        ),
        Tensor("Probabilities", (spec.batch, spec.top_k), "bf16", read_only=True),
        Tensor("Output", (spec.batch, spec.width), "bf16"),
    )
    return (
        ts
        + (
            Tensor(
                "Assignments", (spec.batch * spec.top_k, 3), "int32", read_only=True
            ),
        )
        if spec.grouped
        else ts
    )


def plan(spec):
    from .vector import Body, index

    verify_spec(spec)
    ts = signature(spec)
    b = Body()
    z = IndexExpr()
    row = index(V.ROW)
    if isinstance(spec, RouterSpec):
        width = spec.experts
        rows = spec.batch
        groups = tuple(WorkGroup(bindings=((V.ROW, i),)) for i in range(rows))
        logits = b.cast(
            "logits_fp32", b.load("load_logits", "Logits", row, z, width), width
        )
        maximum = b.math("maximum", M.MAX_REDUCTION, (logits,), reduction=("D",))
        shifted = b.math("shifted", M.SUBTRACT, (logits, maximum), width)
        exponential = b.math("exponential", M.EXP, (shifted,), width)
        denominator = b.math(
            "denominator", M.SUM_REDUCTION, (exponential,), reduction=("D",)
        )
        probabilities = b.math("softmax", M.DIVIDE, (exponential, denominator), width)
        selected, ids = b.topk("topk", probabilities, spec.top_k)
        selected_sum = b.math(
            "selected_sum", M.SUM_REDUCTION, (selected,), reduction=("D",)
        )
        weights = b.math("renormalized", M.DIVIDE, (selected, selected_sum), spec.top_k)
        b.store(
            "store_probs",
            b.cast("Output_probs", weights, spec.top_k, "bf16"),
            "Probabilities",
            row,
            z,
            spec.top_k,
        )
        b.store("store_ids", ids, "ExpertIDs", row, z, spec.top_k)
    elif isinstance(spec, HistogramSpec):
        width = spec.experts
        rows = 1
        groups = (WorkGroup(),)
        ids = b.load(
            "load_ids",
            "ExpertIDs",
            z,
            z,
            spec.top_k,
            "int32",
            height=spec.batch,
            axes=("ROW", "D"),
        )
        histogram = b.math(
            "histogram",
            M.HISTOGRAM,
            (ids,),
            width,
            "int32",
            constants=(("bins", float(width)),),
            reduction=("ROW", "D"),
        )
        prefix = b.math(
            "prefix", M.PREFIX_SUM, (histogram,), width + 1, "int32", reduction=("D",)
        )
        b.store("store_counts", histogram, "Counts", z, z, width)
        b.store("store_offsets", prefix, "Offsets", z, z, width + 1)
    elif isinstance(spec, GatherSpec):
        width = spec.width
        rows = spec.batch * spec.top_k
        ordered = sorted(spec.assignments, key=lambda a: (a.expert, a.token, a.slot))
        ranks = Counter()
        group_list = []
        for output, assignment in enumerate(ordered):
            group_list.append(
                WorkGroup(
                    bindings=(
                        (V.ROW, output),
                        (V.TOKEN, assignment.token),
                        (V.SLOT, assignment.slot),
                        (V.CHANNEL, assignment.expert),
                    ),
                    constants=(
                        ("rank", float(ranks[assignment.expert])),
                        ("token", float(assignment.token)),
                        ("slot", float(assignment.slot)),
                    ),
                )
            )
            ranks[assignment.expert] += 1
        groups = tuple(group_list)
        expert = b.load(
            "expert", "ExpertIDs", index(V.TOKEN), index(V.SLOT), 1, "int32"
        )
        offset = b.load("offset", "Offsets", z, index(V.CHANNEL), 1, "int32")
        rank = b.math("rank", M.PARAMETER, dtype="int32", parameter="rank")
        output_index = b.math("output_index", M.ADD, (offset, rank), dtype="int32")
        token = b.math("token", M.PARAMETER, dtype="int32", parameter="token")
        slot = b.math("slot", M.PARAMETER, dtype="int32", parameter="slot")
        metadata = b.math("metadata", M.PACK_VECTOR, (token, slot, expert), 3, "int32")
        activation = b.load("activation", "X", index(V.TOKEN), z, width)
        b.ops[-1] = replace(
            b.ops[-1],
            dependencies=(
                Dependency("output_index", Phase.COMPLETE, Relation.EXPLICIT),
            ),
        )
        probability = b.load(
            "probability", "Probabilities", index(V.TOKEN), index(V.SLOT), 1
        )
        b.store(
            "store_activation",
            activation,
            "SortedX",
            row,
            z,
            width,
            extra=(output_index,),
        )
        b.store(
            "store_assignment",
            metadata,
            "Assignments",
            row,
            z,
            3,
            extra=(output_index,),
        )
        b.store(
            "store_probability",
            probability,
            "SortedProbabilities",
            row,
            z,
            1,
            extra=(output_index,),
        )
    else:
        width = spec.width
        rows = spec.batch
        groups = tuple(
            WorkGroup(
                bindings=(
                    (V.ROW, token),
                    *((f"assignment_{i}", r) for i, r in enumerate(mapping)),
                )
            )
            for token, mapping in enumerate(spec.rows_by_token)
        )
        accumulator = b.constant("zero", 0)
        for i in range(spec.top_k):
            source = IndexExpr(terms=((f"assignment_{i}", 1),))
            if spec.grouped:
                b.load(f"metadata_{i}", "Assignments", source, z, 3, "int32")
            activation = b.load(f"expert_{i}", "ExpertOutput", source, z, width)
            if spec.grouped:
                b.ops[-1] = replace(
                    b.ops[-1],
                    dependencies=(
                        Dependency(f"metadata_{i}", Phase.COMPLETE, Relation.EXPLICIT),
                    ),
                )
            activation = b.cast(f"expert_fp32_{i}", activation, width)
            probability = b.cast(
                f"probability_fp32_{i}",
                b.load(f"probability_{i}", "Probabilities", row, IndexExpr(i), 1),
                1,
            )
            product = b.math(
                f"weighted_{i}", M.MULTIPLY, (activation, probability), width
            )
            accumulator = b.math(
                f"accumulator_{i}", M.ADD, (accumulator, product), width
            )
        b.store(
            "store_output",
            b.cast("Output", accumulator, width, "bf16"),
            "Output",
            row,
            z,
            width,
        )
    contract = AlgorithmContract(
        (IndexDomain("ROWS", rows), IndexDomain("D", width)),
        tuple(
            OperandContract(t.id, ("ROW", "COL"), t.dtype, not t.read_only) for t in ts
        ),
        tuple(op.math for op in b.ops if op.math is not None),
        output_dtype="int32" if isinstance(spec, HistogramSpec) else "bf16",
    )
    return ts, groups, tuple(b.ops), contract, rows, width
