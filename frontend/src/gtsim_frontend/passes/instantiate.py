from ..kernels.hopper.reference import gemm, fa3


def instantiate(algorithm, num_sms=132):
    algorithm.verify()
    if algorithm.family == "vector":
        from ..kernels.vector import instantiate as vector

        return vector(algorithm, num_sms)
    if algorithm.schedule_id == "decode_combine":
        from ..kernels.hopper.attention_combine import instantiate as combine

        return combine(algorithm, num_sms)
    if algorithm.schedule_id == "decode_simt":
        from ..kernels.hopper.attention_decode import instantiate as attention_decode

        return attention_decode(algorithm, num_sms)
    if algorithm.family == "gemm":
        return gemm.instantiate(algorithm, num_sms)
    return fa3.instantiate(algorithm, num_sms)
