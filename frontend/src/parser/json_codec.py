"""Structural serialization for the canonical frontend AST."""

from dataclasses import fields, is_dataclass
from enum import Enum
import json
from . import ast as A

_RECORDS = {
    name: value
    for name, value in vars(A).items()
    if isinstance(value, type) and is_dataclass(value)
}
_ENUMS = {
    name: value
    for name, value in vars(A).items()
    if isinstance(value, type) and issubclass(value, Enum)
}


def encode(value):
    if isinstance(value, Enum):
        return {"enum": type(value).__name__, "value": value.value}
    if is_dataclass(value):
        return {
            "record": type(value).__name__,
            **{
                field.name: encode(getattr(value, field.name))
                for field in fields(value)
            },
        }
    if isinstance(value, (tuple, list)):
        return [encode(item) for item in value]
    return value


def decode(value):
    if isinstance(value, list):
        return tuple(decode(item) for item in value)
    if not isinstance(value, dict):
        return value
    if "enum" in value:
        if set(value) != {"enum", "value"} or value["enum"] not in _ENUMS:
            raise ValueError("unknown enum or invalid enum fields")
        return _ENUMS[value["enum"]](value["value"])
    if "record" not in value or value["record"] not in _RECORDS:
        raise ValueError("unknown AST record")
    cls = _RECORDS[value["record"]]
    if set(value) - {"record"} != {field.name for field in fields(cls)}:
        raise ValueError(f"{cls.__name__}: missing or unknown fields")
    return cls(**{key: decode(item) for key, item in value.items() if key != "record"})


def dumps(ir):
    return (
        json.dumps(
            encode(ir),
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )


def loads(text):
    result = decode(json.loads(text))
    if not isinstance(
        result, (A.ProgIR, A.AlgoIR, A.FlowIR, A.ArchIR)
    ):
        raise ValueError("expected an IR root record")
    return result
