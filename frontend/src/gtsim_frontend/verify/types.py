"""Enforce typed structured records at API/JSON boundaries, not just hints."""

from dataclasses import fields, is_dataclass
from enum import Enum
from functools import lru_cache
from types import UnionType
from typing import get_args, get_origin, get_type_hints, Union
from ..ir.common import require, VerificationError


@lru_cache(None)
def annotations(cls):
    return get_type_hints(cls)


def verify_types(value, expected=None, location="IR"):
    if expected is None:
        expected = type(value)
    origin = get_origin(expected)
    args = get_args(expected)
    if origin in (Union, UnionType):
        for candidate in args:
            try:
                verify_types(value, candidate, location)
                return
            except VerificationError:
                pass
        require(False, location, f"expected {expected}")
    elif origin is tuple:
        require(type(value) is tuple, location, "expected immutable tuple")
        if len(args) == 2 and args[1] is Ellipsis:
            for i, item in enumerate(value):
                verify_types(item, args[0], f"{location}[{i}]")
        else:
            require(len(value) == len(args), location, "tuple arity mismatch")
            for i, (item, kind) in enumerate(zip(value, args)):
                verify_types(item, kind, f"{location}[{i}]")
    elif isinstance(expected, type) and issubclass(expected, Enum):
        require(
            isinstance(value, expected), location, f"expected typed {expected.__name__}"
        )
    elif is_dataclass(expected):
        require(type(value) is expected, location, f"expected {expected.__name__}")
        hints = annotations(expected)
        for field in fields(value):
            verify_types(
                getattr(value, field.name),
                hints[field.name],
                f"{location}.{field.name}",
            )
    else:
        require(type(value) is expected, location, f"expected {expected.__name__}")
