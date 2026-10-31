"""Select Hopper patterns without expanding the FA3 loop domain."""

from .hopper.lower import lower as lower_patterns


def lower(flow):
    if flow.family != "fa3":
        raise ValueError("expected fa3 Flow")
    return lower_patterns(flow)
