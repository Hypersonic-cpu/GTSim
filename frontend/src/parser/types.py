"""Validate typed AST fields at the parser boundary."""

from dataclasses import fields, is_dataclass
from types import UnionType
from typing import get_args, get_origin, get_type_hints, Union
from . import ast as A


def check(value, expected=None, location="IR"):
    expected = type(value) if expected is None else expected
    origin, args = get_origin(expected), get_args(expected)
    if origin in (Union, UnionType):
        for choice in args:
            try:
                check(value, choice, location)
                return
            except ValueError:
                pass
        raise ValueError(f"{location}: expected {expected}")
    if origin is tuple:
        if type(value) is not tuple:
            raise ValueError(f"{location}: expected tuple")
        if len(args) == 2 and args[1] is Ellipsis:
            for index, item in enumerate(value):
                check(item, args[0], f"{location}[{index}]")
        elif len(args) != len(value):
            raise ValueError(f"{location}: tuple arity mismatch")
        else:
            for index, (item, kind) in enumerate(zip(value, args)):
                check(item, kind, f"{location}[{index}]")
    elif type(value) is not expected:
        raise ValueError(f"{location}: expected {expected.__name__}")
    elif is_dataclass(value):
        hints = get_type_hints(type(value))
        for field in fields(value):
            check(
                getattr(value, field.name),
                hints[field.name],
                f"{location}.{field.name}",
            )
        if isinstance(value, A.MathOperation):
            subtypes = {
                A.OperationClass.ARITHMETIC: A.ArithmeticCost,
                A.OperationClass.SPECIAL_FUNCTION: A.SpecialFunction,
                A.OperationClass.REDUCTION: A.ReductionKind,
                A.OperationClass.COLLECTIVE: A.CollectiveKind,
            }
            cost_type = subtypes.get(value.category, type(None))
            if type(value.cost) is not cost_type:
                raise ValueError(
                    f"{location}: incompatible operation category and cost subtype"
                )
