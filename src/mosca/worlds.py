from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable

from .compact2_world import Compact2World
from .factorized_world import FactorizedWorld
from .compact_world import CompactWorld
from .motor_env import MotorMaze, OracleCounter
from .native_world import NativeWorld
from .tasks import Task


@runtime_checkable
class ProgramWorld(Protocol):
    """Minimal contract a programming language/world must expose to Mosca."""

    task: Task
    max_steps: int
    done: bool
    last_evaluation: object | None
    actions: list[str]

    def reset(self) -> dict: ...
    def observe(self) -> dict: ...
    def valid_actions(self) -> tuple[str, ...]: ...
    def step(self, action: str) -> tuple[dict, float, bool, dict]: ...
    def evaluate_hidden(self): ...
    def probe_score(self) -> float: ...
    def source(self) -> str: ...


WorldFactory = Callable[[Task, int, OracleCounter | None], ProgramWorld]


def _python_factory(
    task: Task,
    max_steps: int = 8,
    counter: OracleCounter | None = None,
) -> ProgramWorld:
    return MotorMaze(task, max_steps=max_steps, counter=counter)


def _mosca_factory(
    task: Task,
    max_steps: int = 8,
    counter: OracleCounter | None = None,
) -> ProgramWorld:
    return NativeWorld(task, max_steps=max_steps, counter=counter)


def _compact_factory(
    task: Task,
    max_steps: int = 8,
    counter: OracleCounter | None = None,
) -> ProgramWorld:
    return CompactWorld(task, max_steps=max_steps, counter=counter)


def _compact2_factory(
    task: Task,
    max_steps: int = 8,
    counter: OracleCounter | None = None,
) -> ProgramWorld:
    return Compact2World(task, max_steps=max_steps, counter=counter)


def _factorized_factory(
    task: Task,
    max_steps: int = 8,
    counter: OracleCounter | None = None,
) -> ProgramWorld:
    return FactorizedWorld(task, max_steps=max_steps, counter=counter)


_WORLD_FACTORIES: dict[str, WorldFactory] = {
    "compact": _compact_factory,
    "compact2": _compact2_factory,
    "factorized": _factorized_factory,
    "mosca": _mosca_factory,
    "python": _python_factory,
}


def register_world(name: str, factory: WorldFactory, *, replace: bool = False) -> None:
    """Register another language world without changing Mosca's learner."""

    key = name.strip().lower()
    if not key:
        raise ValueError("world name cannot be empty")
    if key in _WORLD_FACTORIES and not replace:
        raise ValueError(f"world {key!r} is already registered")
    _WORLD_FACTORIES[key] = factory


def world_names() -> tuple[str, ...]:
    return tuple(sorted(_WORLD_FACTORIES))


def create_world(
    name: str,
    task: Task,
    *,
    max_steps: int = 8,
    counter: OracleCounter | None = None,
) -> ProgramWorld:
    key = name.strip().lower()
    try:
        factory = _WORLD_FACTORIES[key]
    except KeyError as exc:
        raise ValueError(
            f"unknown world {name!r}; available={world_names()}"
        ) from exc
    world = factory(task, max_steps, counter)
    if not isinstance(world, ProgramWorld):
        raise TypeError(f"world {key!r} does not implement ProgramWorld")
    return world
