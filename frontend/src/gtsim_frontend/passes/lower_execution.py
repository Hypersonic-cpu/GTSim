from ..ir.execution import ExecKernel
from ..serialization.json_codec import digest


def lower(arch):
    arch.verify()
    data = dict(vars(arch))
    data["source_hash"] = digest(arch)
    kernel = ExecKernel(**data)
    kernel.verify()
    return kernel
