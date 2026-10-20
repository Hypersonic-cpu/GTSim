"""Public GEMM DAG lowering entry point.

Compiler-generated Hopper IR uses operation-driven synchronization lowering.
The old strict recognizer is retained only for explicitly tagged reference
kernels and their historical parity tests, not for scheduling or search.
"""


def lower_gemm(tvm, func, hardware=None):
    if func.attrs and func.attrs.get('gtsim.compiler')=='hopper_intrinsics_v1':
        if hardware is None:
            raise ValueError('native kernel lowering requires the simulation hardware profile')
        from native_dag import lower_native
        return lower_native(tvm,func,hardware)
    if func.attrs and func.attrs.get('gtsim.schedule')=='h100_persistent_v1':
        from reference_gemm_lower import lower_reference_gemm
        return lower_reference_gemm(tvm,func)
    raise ValueError('unsupported hardware operations or missing compiler buffer-layout ABI')
