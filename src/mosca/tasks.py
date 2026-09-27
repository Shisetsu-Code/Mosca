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
    hidden_cases: tuple[Case, ...] = ()


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
        hidden_cases=(
            Case(([2, 4, 6, 8],), 20),
            Case(([-10, -20, 5],), -25),
            Case(([1000],), 1000),
            Case(([0, 0, 0],), 0),
            Case(([12, -7, -5, 1],), 1),
            Case(([3, -3, 3, -3, 3],), 3),
            Case(([-1, -1, -1, -1],), -4),
            Case(([50, 25, -75, 1],), 1),
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
        hidden_cases=(
            Case(([-5, 0, 5, 10, -10],), 2),
            Case(([100, 1, 1, -1],), 3),
            Case(([0, 0, 0, 0],), 0),
            Case(([-1, -2, -3],), 0),
            Case(([1],), 1),
            Case(([1, -1, 1, -1, 1],), 3),
            Case(([999, -999, 0],), 1),
            Case(([2, 2, 2, 2, 2],), 5),
            Case(([-2, 3, -4, 5, 6],), 3),
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
        hidden_cases=(
            Case(([10, 9, 8, 7],), 10),
            Case(([-100, -50, -75],), -50),
            Case(([0, 0, 0],), 0),
            Case(([999, -1, 500],), 999),
            Case(([1, 100, 2, 99],), 100),
            Case(([-1],), -1),
            Case(([5, 5, 5],), 5),
            Case(([-5, 0, -1],), 0),
        ),
    ),
}
