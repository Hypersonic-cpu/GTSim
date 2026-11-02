"""Shared scheduled SIMD, reduction, SFU and memory instruction selection."""

from parser import ast as A
from .hopper import index, dimension


def select_vector(op, flow):
    steps = []
    compute_executor = (
        index("%executor")
        if any(loop.variable == "executor" for loop in op.iterations)
        else index(0)
    )
    if op.kind in (A.OpKind.COPY, A.OpKind.STORE):
        shape = op.region.shape
        leading = None  # Filled from the declaration by the target-region pass.
        dtype = op.results[0].dtype if op.results else op.cast
        dependencies = op.dependencies
        if op.kind == A.OpKind.STORE:
            steps.append(
                A.NativeStep(
                    A.NativeOpcode.SHARED_STORE,
                    A.ExecutionResource.STORE,
                    compute_executor,
                    op.iterations,
                    op.region,
                    dtype,
                    None,
                    dependencies,
                    (),
                    (),
                    op.guard,
                    op.static_args,
                )
            )
            opcode, resource, dependencies, inputs = (
                A.NativeOpcode.TMA_STORE,
                A.ExecutionResource.STORE,
                (),
                (0,),
            )
        else:
            opcode, resource, inputs = (
                A.NativeOpcode.TMA_LOAD,
                A.ExecutionResource.LOAD,
                (),
            )
        steps.append(
            A.NativeStep(
                opcode,
                resource,
                index(flow.config.producer_warp),
                op.iterations,
                op.region,
                dtype,
                leading,
                dependencies,
                inputs,
                (),
                op.guard,
                op.static_args,
            )
        )
    else:
        math = op.math
        opcode, resource = A.NativeOpcode.ADD, A.ExecutionResource.ARITHMETIC
        if math.category == A.OperationClass.COLLECTIVE:
            opcode, resource = A.NativeOpcode.SHUFFLE, A.ExecutionResource.SHUFFLE
        elif not op.operands and math.category == A.OperationClass.ARITHMETIC:
            opcode = A.NativeOpcode.FILL
        elif math.category == A.OperationClass.SPECIAL_FUNCTION:
            opcode = {
                A.SpecialFunction.EXPONENTIAL: A.NativeOpcode.EXP,
                A.SpecialFunction.LOGARITHMIC: A.NativeOpcode.LOG,
                A.SpecialFunction.ROOT: A.NativeOpcode.RSQRT,
                A.SpecialFunction.TRIGONOMETRIC: A.NativeOpcode.SIN,
            }[math.cost]
            resource = A.ExecutionResource.SPECIAL_FUNCTION
        elif math.category == A.OperationClass.REDUCTION:
            opcode = (
                A.NativeOpcode.REDUCE_SUM
                if math.cost == A.ReductionKind.SUM
                else A.NativeOpcode.REDUCE_MAX
            )
        elif math.category == A.OperationClass.CONVERT:
            opcode = A.NativeOpcode.CAST
        elif math.cost == A.ArithmeticCost.MULTIPLY:
            opcode = A.NativeOpcode.MUL
        elif math.cost == A.ArithmeticCost.DIVIDE:
            opcode, resource = (
                A.NativeOpcode.DIVIDE,
                A.ExecutionResource.SPECIAL_FUNCTION,
            )
        steps.append(
            A.NativeStep(
                opcode,
                resource,
                compute_executor,
                op.iterations,
                None,
                op.results[0].dtype,
                None,
                op.dependencies,
                (),
                op.work_shape or op.results[0].shape,
                op.guard,
                op.static_args,
            )
        )
    return A.NativePattern(op.id, op.scope, tuple(steps), op.dependencies)
