"""IR-03 → IR-04: select target patterns without resolving macros or loops."""

from dataclasses import replace
from parser import ast as A
from .static import EMPTY
from .nvidia.hopper import lower_region


def lower_flow(flow):
    return A.ArchIR(
        flow.symbol,
        flow.static,
        flow.tensors,
        tuple(
            lower_region(
                replace(region, static=replace(flow.static, arguments=()))
                if region.static == EMPTY
                else region
            )
            for region in flow.regions
        ),
        flow.stream,
        flow.outputs,
    )
