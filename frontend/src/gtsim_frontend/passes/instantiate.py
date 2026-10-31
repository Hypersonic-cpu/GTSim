from ..kernels.hopper.reference import gemm, fa3


def instantiate(algorithm, num_sms=132):
    algorithm.verify()
    if algorithm.family=='gemm':return gemm.instantiate(algorithm,num_sms)
    return fa3.instantiate(algorithm,num_sms)
