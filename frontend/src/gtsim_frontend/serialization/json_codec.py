"""Typed JSON codec: reject unknown records and fields; no executable payloads."""

import dataclasses
import hashlib
import json
import math
from enum import Enum
from ..ir import (
    common,
    algorithm,
    program,
    mapping,
    flow,
    arch,
    execution,
    profiles,
    schedule,
    graph,
    decode,
)

_TYPES = {
    c.__name__: c
    for module in (
        common,
        algorithm,
        program,
        mapping,
        flow,
        arch,
        execution,
        profiles,
        schedule,
        graph,
        decode,
    )
    for c in vars(module).values()
    if isinstance(c, type) and dataclasses.is_dataclass(c)
}
_FORMATS = {
    "Program": "prog-ir/0.1",
    "Algorithm": "algo-ir/0.2",
    "FlowKernel": "flow-ir/0.2",
    "ArchKernel": "arch-ir/0.2",
    "ExecKernel": "gtsim.exec.kernel/2",
    "ExecProgram": "gtsim.exec.program/1",
    "DecodeProgram": "prog-ir/0.2",
    "DecodeAlgorithm": "algo-ir/0.3",
    "DecodeFlowKernel": "flow-ir/0.3",
    "DecodeNativeKernel": "arch-ir/0.3",
}


_ENUMS = {
    c.__name__: c
    for module in (algorithm, schedule, decode)
    for c in vars(module).values()
    if isinstance(c, type) and issubclass(c, Enum) and c is not Enum
}


def encode(value):
    if isinstance(value, Enum):
        return {"enum": type(value).__name__, "value": value.value}
    if dataclasses.is_dataclass(value):
        return {
            "record": type(value).__name__,
            **{
                f.name: encode(getattr(value, f.name))
                for f in dataclasses.fields(value)
            },
        }
    if isinstance(value, tuple):
        return [encode(v) for v in value]
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("non-finite float in IR")
        return value
    if value is None or type(value) in (str, int, bool):
        return value
    raise ValueError(f"unsupported IR value: {type(value).__name__}")


def decode(value):
    if isinstance(value, list):
        return tuple(decode(v) for v in value)
    if isinstance(value, dict):
        if set(value) == {"enum", "value"}:
            if value["enum"] not in _ENUMS:
                raise ValueError("unknown typed enum")
            return _ENUMS[value["enum"]](value["value"])
        name = value.get("record")
        if name not in _TYPES:
            raise ValueError(f"unknown record: {name}")
        cls = _TYPES[name]
        expected = {f.name for f in dataclasses.fields(cls)}
        if set(value) != expected | {"record"}:
            raise ValueError(f"{name}: missing or unknown fields")
        return cls(**{k: decode(v) for k, v in value.items() if k != "record"})
    return value


def dumps(value):
    value.verify()
    return (
        json.dumps(
            {"format": _FORMATS[type(value).__name__], **encode(value)},
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )


def loads(text):
    data = json.loads(text)
    if not isinstance(data, dict) or "format" not in data:
        raise ValueError("missing format discriminator")
    format_name = data.pop("format")
    obj = decode(data)
    if _FORMATS.get(type(obj).__name__) != format_name:
        raise ValueError("format discriminator mismatch")
    obj.verify()
    return obj


def digest(value):
    return hashlib.sha256(dumps(value).encode()).hexdigest()
