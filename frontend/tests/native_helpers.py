from dataclasses import replace

from parser import ast as A
from compile.expand_native import NativeExpander
from compile.static import EMPTY, Resolver, bound_resolver
from compile.streaming import _strides


def expand_arch(arch):
    root = bound_resolver(arch.static, Resolver(EMPTY, {})).materialize_constants()
    tensors = {}
    tensor_records = []
    for spec in arch.tensors:
        shape = tuple(root.value(d) for d in spec.shape)
        tensor = A.KernelTensor(
            len(tensor_records), spec.id, shape, _strides(shape), spec.dtype,
            spec.tier or A.MemorySpace.HBM, spec.external, spec.read_only, None,
        )
        tensors[spec.id] = tensor
        tensor_records.append(tensor)
    all_ctas, all_ops = [], []
    for region in arch.regions:
        arguments = {a.id: root.value(a.expression) for a in region.static.arguments}
        resolver = Resolver(replace(region.static, parameters=(), constraints=()),
                            {**root.values, **arguments}).materialize_constants()
        NativeExpander(region, resolver, tensors).expand_to(all_ctas.append, all_ops.append)
    return tensor_records, all_ctas, all_ops


def expand_region(arch):
    tensors, ctas, ops = expand_arch(arch)
    return tensors, ctas, ops
