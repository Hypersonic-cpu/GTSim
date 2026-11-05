"""Load the sibling AST under a private namespace without shadowing stdlib ast."""

from importlib import import_module
from pathlib import Path
import sys
from types import ModuleType

_NAMESPACE = "gtsim_text_ast"
if _NAMESPACE not in sys.modules:
    package = ModuleType(_NAMESPACE)
    package.__path__ = [str(Path(__file__).resolve().parents[1] / "ast")]
    sys.modules[_NAMESPACE] = package

for _name in (
    "model",
    "common",
    "algorithm",
    "program",
    "mapping",
    "profiles",
    "schedule",
    "flow",
    "layout",
    "graph",
    "arch",
    "execution",
):
    _module = import_module(f"{_NAMESPACE}.{_name}")
    for _key, _value in vars(_module).items():
        if isinstance(_value, type) and _value.__module__ == _module.__name__:
            globals()[_key] = _value

from gtsim_text_ast.model import Node
