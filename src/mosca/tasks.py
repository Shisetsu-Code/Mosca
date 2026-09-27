from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Case:
    args: tuple[Any, ...]
    expected: Any


@dataclass(frozen=True)
class Task:
    name: str
    description: str
    cases: tuple[Case, ...]


TASKS: dict[str, Task] = {
    "sum_list": Task(
        name="sum_list",
        description="Return the sum of all integers in xs.",
        cases=(
            Case(([1, 2, 3],), 6),
            Case(([-4, 1, 5],), 2),
            Case(([],), 0),
            Case(([7],), 7),
        ),
    ),
    "count_positive": Task(
        name="count_positive",
        description="Count values in xs strictly greater than zero.",
        cases=(
            Case(([-2, -1, 0, 1, 2],), 2),
            Case(([1, 2, 3],), 3),
            Case(([-1, 0],), 0),
            Case(([],), 0),
            Case(([5, -1, 6, -2],), 2),
            Case(([1, 2, -9],), 2),
            Case(([0, 0, 7, 0],), 1),
            Case(([-5, 8, -3, 2, -1],), 2),
            Case(([1, 1, 1],), 3),
            Case(([10, 1, 1],), 3),
            Case(([2, 2],), 2),
        ),
    ),
    "max_list": Task(
        name="max_list",
        description="Return the largest integer in a non-empty xs.",
        cases=(
            Case(([1, 2, 3],), 3),
            Case(([5, -1, 4],), 5),
            Case(([-8, -2, -10],), -2),
            Case(([9],), 9),
        ),
    ),
}
