"""Independent logical BM128 BN256 BK64 two-stage GEMM schedule."""

from ....ir.common import Tensor, TensorRegion
from ....ir.flow import FlowKernel
from ....ir.mapping import Mapping, Executor, Buffer
from ....serialization.json_codec import digest
from .builder import FlowBuilder


def instantiate(algorithm, num_sms=132):
    algorithm.verify()
    if algorithm.family != "gemm":
        raise ValueError("GEMM schedule requires GEMM algorithm")
    m, n, k = algorithm.dimensions
    tensors = (
        Tensor("A", (m, k), read_only=True),
        Tensor("B", (k, n), read_only=True),
        Tensor("C", (m, n)),
    )
    if type(num_sms) is not int or num_sms < 1:
        raise ValueError("invalid SM count")
    grid = min(num_sms, (m // 128) * (n // 256))
    builder = FlowBuilder()
    tile = 0
    # Original grouped-column output order. Persistent CTA assignment belongs
    # to native lowering and does not alter this logical coverage.
    columns = n // 256
    for start in range(0, columns, min(8, columns)):
        for row in range(m // 128):
            for col in range(start, min(columns, start + min(8, columns))):
                prefix = f"cta{tile % grid}.tile{tile}"
                producer = f"cta{tile % grid}.producer"
                matrix = f"cta{tile % grid}.matrix"
                buffer = f"cta{tile % grid}.AB"
                builder.producer = producer
                acc_event, acc = builder.emit(
                    prefix + ".zero",
                    "compute",
                    matrix,
                    semantic="zero_accumulator",
                    shape=(128, 256),
                )
                for iteration in range(k // 64):
                    p = f"{prefix}.k{iteration}"
                    slot, owned = builder.acquire(p + ".acquire", buffer, iteration % 2)
                    a_event, a = builder.emit(
                        p + ".A",
                        "load_async",
                        producer,
                        inputs=(owned,),
                        slot=slot,
                        reads=(
                            TensorRegion("A", (row * 128, iteration * 64), (128, 64)),
                        ),
                        shape=(128, 64),
                        dtype="fp16",
                    )
                    b_event, b = builder.emit(
                        p + ".B",
                        "load_async",
                        producer,
                        inputs=(owned,),
                        slot=slot,
                        reads=(
                            TensorRegion("B", (iteration * 64, col * 256), (64, 256)),
                        ),
                        shape=(64, 256),
                        dtype="fp16",
                    )
                    acc_event, acc = builder.emit(
                        p + ".matmul",
                        "matmul",
                        matrix,
                        inputs=(a_event, b_event, acc_event),
                        slot=slot,
                        semantic="fp16_matmul_fp32_accumulate",
                        operands=(a, b, acc),
                        shape=(128, 256),
                    )
                    builder.release(p + ".release", slot, (acc_event,))
                builder.emit(
                    prefix + ".store",
                    "store_async",
                    producer,
                    inputs=(acc_event,),
                    operands=(acc,),
                    writes=(TensorRegion("C", (row * 128, col * 256), (128, 256)),),
                    semantic="cast_fp16_store",
                )
                tile += 1
    flow = FlowKernel(
        symbol=algorithm.symbol,
        source_hash=digest(algorithm),
        family="gemm",
        schedule_id="ref_h100_gemm_v0_1_0",
        tensors=tensors,
        executors=tuple(
            e
            for c in range(grid)
            for e in (
                Executor(f"cta{c}.producer", "producer", (8,)),
                Executor(f"cta{c}.matrix", "matrix", tuple(range(8))),
            )
        ),
        buffers=tuple(
            Buffer(f"cta{c}.AB", 2, 2 * 64 * (128 + 256)) for c in range(grid)
        ),
        mapping=Mapping(
            ctas=tuple((c, c) for c in range(grid)),
            execute=tuple(
                x
                for c in range(grid)
                for x in ((f"cta{c}.producer", "TMA"), (f"cta{c}.matrix", "WGMMA"))
            ),
            store=tuple((f"cta{c}.AB", "SMEM") for c in range(grid)),
        ),
        ops=tuple(builder.ops),
    )
    flow.verify()
    return flow
