from ..kernels.hopper.reference import gemm, fa3


def instantiate(algorithm, num_sms=132):
    algorithm.verify()
    from ..ir.decode import DecodeAlgorithm

    if isinstance(algorithm, DecodeAlgorithm):
        from .decode_lowering import make_flow

        return make_flow(algorithm)
    if algorithm.family == "gemm":
        return gemm.instantiate(algorithm, num_sms)
    return fa3.instantiate(algorithm, num_sms)
