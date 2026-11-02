"""Program invocations reference definitions rather than embedding algorithms."""

from dataclasses import dataclass
from enum import Enum
from .common import Header, Tensor, StaticIf


class CommandKind(str, Enum):
    LAUNCH = "launch"
    SYNCHRONIZE = "synchronize"
    RECORD_EVENT = "record_event"
    WAIT_EVENT = "wait_event"


@dataclass(frozen=True)
class Invocation:
    id: str
    callee: str
    implementation: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    events: tuple[str, ...]
    attributes: tuple[tuple[str, str], ...]
    guard: StaticIf | None


@dataclass(frozen=True)
class Command:
    kind: CommandKind
    target: str
    after: tuple[str, ...]


@dataclass(frozen=True)
class ProgIR(Header):
    streams: tuple[str, ...]
    tensors: tuple[Tensor, ...]
    launches: tuple[Invocation, ...]
    commands: tuple[Command, ...]
    returns: tuple[str, ...]
    return_after: tuple[str, ...]
