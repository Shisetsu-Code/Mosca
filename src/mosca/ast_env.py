from __future__ import annotations

import warnings
from dataclasses import dataclass

from .grammar import (
    PRODUCTIONS, count_holes, count_nodes, expand, first_hole, initial_tree,
    replace_first_hole, source, tag_counts, to_ast,
)
from .tasks import Task


@dataclass(frozen=True)
class Evaluation:
    compiled: bool
    passed: int
    total: int
    error: str | None = None

    @property
    def score(self) -> float:
        return self.passed / self.total if self.total else 0.0


class ASTMaze:
    """Generic typed-hole program-construction environment."""

    COMPLEX_ACTIONS = {
        "LIST:CONS", "STMT:FOR", "STMT:IF",
        "EXPR:BINOP", "EXPR:COMPARE", "EXPR:SUBSCRIPT",
    }

    def __init__(self, task: Task, max_steps: int = 40):
        self.task = task
        self.max_steps = max_steps
        self.reset()

    def reset(self) -> dict:
        self.tree = initial_tree()
        self.actions: list[str] = []
        self.done = False
        self.last_evaluation: Evaluation | None = None
        return self.observe()

    def valid_actions(self) -> tuple[str, ...]:
        if self.done:
            return ()
        hole = first_hole(self.tree)
        if hole is None:
            return ()
        actions = list(PRODUCTIONS[hole.kind])
        if self.max_steps - len(self.actions) <= 5:
            pruned = [a for a in actions if a not in self.COMPLEX_ACTIONS]
            if pruned:
                actions = pruned
        return tuple(actions)

    def observe(self) -> dict:
        hole = first_hole(self.tree)
        return {
            "task": self.task.name,
            "first_hole": hole.kind if hole else None,
            "holes": count_holes(self.tree),
            "nodes": count_nodes(self.tree),
            "tags": tag_counts(self.tree),
            "steps": len(self.actions),
            "valid_actions": list(self.valid_actions()) if not self.done else [],
        }

    def step(self, action: str) -> tuple[dict, float, bool, dict]:
        if action not in self.valid_actions():
            raise ValueError(f"invalid action {action}; valid={self.valid_actions()}")
        hole = first_hole(self.tree)
        assert hole is not None
        self.tree, changed = replace_first_hole(self.tree, expand(hole.kind, action))
        assert changed
        self.actions.append(action)
        reward = -0.01
        info: dict = {}

        if count_holes(self.tree) == 0:
            self.last_evaluation = self.evaluate()
            self.done = True
            reward += 0.25 if self.last_evaluation.compiled else -0.25
            reward += self.last_evaluation.score * 5.0
            if self.last_evaluation.score == 1.0:
                reward += 5.0
            info["evaluation"] = self.last_evaluation
        elif len(self.actions) >= self.max_steps:
            self.done = True
            reward -= 0.5

        return self.observe(), reward, self.done, info

    def _evaluate_cases(self, cases) -> Evaluation:
        try:
            module = to_ast(self.tree)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                code = compile(module, "<mosca-ast>", "exec", dont_inherit=True, optimize=0)
            namespace: dict = {"__builtins__": {}}
            exec(code, namespace, namespace)
            solve = namespace["solve"]
        except Exception as exc:
            return Evaluation(False, 0, len(cases), f"{type(exc).__name__}: {exc}")

        passed = 0
        first_error: str | None = None
        for case in cases:
            try:
                result = solve(*case.args)
                if result == case.expected:
                    passed += 1
                elif first_error is None:
                    first_error = f"expected {case.expected!r}, got {result!r}"
            except Exception as exc:
                if first_error is None:
                    first_error = f"{type(exc).__name__}: {exc}"
        return Evaluation(True, passed, len(cases), first_error)

    def evaluate(self) -> Evaluation:
        return self._evaluate_cases(self.task.cases)

    def evaluate_hidden(self) -> Evaluation:
        return self._evaluate_cases(self.task.hidden_cases or self.task.cases)

    def source(self) -> str | None:
        return source(self.tree) if count_holes(self.tree) == 0 else None
