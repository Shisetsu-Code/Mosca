from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .motor_env import Evaluation, OracleCounter
from .sensory import task_memory_concepts, task_sensory_tokens
from .tasks import Task


@dataclass
class NativeFrame:
    kind: str
    body: list[Any]


@dataclass(frozen=True)
class Init:
    expr: str


@dataclass
class Loop:
    body: list[Any]


@dataclass(frozen=True)
class Update:
    op: str


@dataclass(frozen=True)
class When:
    condition: str
    op: str


@dataclass(frozen=True)
class Return:
    expr: str = "acc"


class NativeWorld:
    """Direct semantic backend for Mosca's current motor language.

    This world deliberately does not construct Python ASTs and does not call
    compile(). It interprets the same motor actions over a tiny native IR so the
    learner/search stack can be compared against CPython with the world as the
    only changed component.
    """

    CMPS = (">", "<", "==")

    def __init__(
        self,
        task: Task,
        max_steps: int = 12,
        counter: OracleCounter | None = None,
    ):
        self.task = task
        self.max_steps = max_steps
        self.counter = counter if counter is not None else OracleCounter()
        self.sensory = task_sensory_tokens(task)
        self.memory_concepts = task_memory_concepts(task)
        self.reset()

    def reset(self) -> dict:
        self.root: list[Any] = []
        self.stack: list[NativeFrame] = [NativeFrame("function", self.root)]
        self.acc_defined = False
        self.actions: list[str] = []
        self.done = False
        self.last_evaluation: Evaluation | None = None
        self._probe_cache: float | None = None
        return self.observe()

    @property
    def frame(self) -> NativeFrame:
        return self.stack[-1]

    @property
    def in_loop(self) -> bool:
        return any(frame.kind == "for" for frame in self.stack)

    @property
    def in_if(self) -> bool:
        return any(frame.kind == "if" for frame in self.stack)

    def conditions(self) -> list[str]:
        if not self.in_loop:
            return []
        rights = ["0"]
        if self.acc_defined:
            rights.append("acc")
        return [
            f"x{op}{right}"
            for right in rights
            for op in self.CMPS
            if right != "x"
        ]

    def valid_actions(self) -> tuple[str, ...]:
        if self.done:
            return ()

        if self.acc_defined and self.probe_score() >= 1.0 - 1e-12:
            if len(self.stack) > 1:
                return ("END",)
            return ("RETURN:acc",)

        actions: list[str] = []
        if not self.acc_defined:
            if len(self.stack) == 1:
                return tuple(
                    f"SET:acc={expr}"
                    for expr in ("-1", "0", "1", "xs[0]")
                )
            return ()

        if len(self.stack) == 1:
            actions.extend(("FOR:x:xs", "RETURN:acc"))
        elif self.frame.kind == "for":
            actions.extend(("SET:acc=x", "AUG:acc+=x", "AUG:acc+=1"))
            for cond in self.conditions():
                actions.extend((
                    f"WHEN:{cond}:SETX",
                    f"WHEN:{cond}:INC1",
                    f"WHEN:{cond}:ADDX",
                ))
            if self.frame.body:
                actions.append("END")
        elif self.frame.kind == "if":
            actions.extend(("SET:acc=x", "AUG:acc+=x", "AUG:acc+=1"))
            if self.frame.body:
                actions.append("END")

        remaining = self.max_steps - len(self.actions)
        if remaining <= 2:
            closers = [
                action
                for action in actions
                if action == "END" or action.startswith("RETURN:")
            ]
            if closers:
                actions = closers
        return tuple(dict.fromkeys(actions))

    def observe(self) -> dict:
        return {
            "first_hole": "motor",
            "sensory": self.sensory,
            "memory_concepts": self.memory_concepts,
            "holes": max(0, len(self.stack) - 1),
            "nodes": sum(len(frame.body) for frame in self.stack),
            "tags": {
                "frame:function": int(len(self.stack) == 1),
                "frame:for": int(self.in_loop),
                "frame:if": int(self.in_if),
                "acc": int(self.acc_defined),
                "depth": len(self.stack) - 1,
            },
            "steps": len(self.actions),
            "valid_actions": (
                list(self.valid_actions()) if not self.done else []
            ),
        }

    @staticmethod
    def _value(expr: str, xs: list[int], x: int | None, acc: int | None) -> int:
        if expr in {"-1", "0", "1"}:
            return int(expr)
        if expr == "x":
            if x is None:
                raise RuntimeError("x is unavailable outside a loop")
            return x
        if expr == "acc":
            if acc is None:
                raise RuntimeError("acc is undefined")
            return acc
        if expr == "xs[0]":
            return xs[0]
        raise ValueError(f"unsupported native expression {expr!r}")

    @classmethod
    def _condition(
        cls,
        text: str,
        xs: list[int],
        x: int,
        acc: int,
    ) -> bool:
        for symbol in ("==", ">", "<"):
            if symbol in text:
                left, right = text.split(symbol, 1)
                lhs = cls._value(left, xs, x, acc)
                rhs = cls._value(right, xs, x, acc)
                if symbol == "==":
                    return lhs == rhs
                if symbol == ">":
                    return lhs > rhs
                return lhs < rhs
        raise ValueError(f"unsupported native condition {text!r}")

    @classmethod
    def _apply_update(
        cls,
        op: str,
        xs: list[int],
        x: int,
        acc: int,
    ) -> int:
        if op == "SETX":
            return x
        if op == "INC1":
            return acc + 1
        if op == "ADDX":
            return acc + x
        if op == "SETX_DIRECT":
            return x
        raise ValueError(f"unsupported native update {op!r}")

    def _append(self, stmt: Any) -> None:
        self.frame.body.append(stmt)

    def probe_score(self) -> float:
        if not self.acc_defined:
            return 0.0
        if self._probe_cache is None:
            self._probe_cache = self._evaluate_cases(
                self.task.cases,
                hidden=False,
                implicit_return=True,
            ).score
        return self._probe_cache

    def step(self, action: str) -> tuple[dict, float, bool, dict]:
        if action not in self.valid_actions():
            raise ValueError(f"invalid action {action}")

        before_probe = self.probe_score()
        self.actions.append(action)
        reward = -0.02
        info: dict = {}

        if action.startswith("SET:acc="):
            expr = action.split("=", 1)[1]
            if expr == "x":
                self._append(Update("SETX_DIRECT"))
            else:
                self._append(Init(expr))
            self.acc_defined = True
        elif action == "AUG:acc+=x":
            self._append(Update("ADDX"))
        elif action == "AUG:acc+=1":
            self._append(Update("INC1"))
        elif action == "FOR:x:xs":
            loop = Loop([])
            self._append(loop)
            self.stack.append(NativeFrame("for", loop.body))
        elif action.startswith("WHEN:"):
            payload = action[len("WHEN:"):]
            cond, op = payload.rsplit(":", 1)
            self._append(When(cond, op))
        elif action == "END":
            self.stack.pop()
        elif action.startswith("RETURN:"):
            self._append(Return(action.split(":", 1)[1]))
            self.last_evaluation = self.evaluate()
            self.done = True
            reward += self.last_evaluation.score * 5.0
            if self.last_evaluation.score == 1.0:
                reward += 5.0
            info["evaluation"] = self.last_evaluation

        self._probe_cache = None
        after_probe = (
            self.last_evaluation.score
            if self.done and self.last_evaluation
            else self.probe_score()
        )
        reward += 2.0 * (after_probe - before_probe)
        if before_probe >= 1.0 - 1e-12:
            reward += (
                0.30
                if action == "END" or action.startswith("RETURN:")
                else -0.30
            )
        elif after_probe >= 1.0 - 1e-12:
            reward += 0.50

        info["probe_before"] = before_probe
        info["probe_after"] = after_probe

        if len(self.actions) >= self.max_steps and not self.done:
            self.done = True
            reward -= 0.5

        return self.observe(), reward, self.done, info

    def _execute(self, xs: list[int], *, implicit_return: bool) -> int | None:
        acc: int | None = None

        for stmt in self.root:
            if isinstance(stmt, Init):
                acc = self._value(stmt.expr, xs, None, acc)
            elif isinstance(stmt, Loop):
                if acc is None:
                    raise RuntimeError("loop entered before acc initialization")
                for x in xs:
                    for op in stmt.body:
                        if isinstance(op, Update):
                            acc = self._apply_update(op.op, xs, x, acc)
                        elif isinstance(op, When):
                            if self._condition(op.condition, xs, x, acc):
                                acc = self._apply_update(op.op, xs, x, acc)
                        else:
                            raise TypeError(
                                f"unsupported native loop statement {type(op).__name__}"
                            )
            elif isinstance(stmt, Return):
                return self._value(stmt.expr, xs, None, acc)
            else:
                raise TypeError(
                    f"unsupported native statement {type(stmt).__name__}"
                )

        if implicit_return:
            return self._value("acc", xs, None, acc)
        return None

    def _evaluate_cases(
        self,
        cases,
        *,
        hidden: bool,
        implicit_return: bool = False,
    ) -> Evaluation:
        if hidden:
            self.counter.hidden_oracle_calls += 1
        else:
            self.counter.visible_oracle_calls += 1

        passed = 0
        first_error: str | None = None
        for case in cases:
            if hidden:
                self.counter.hidden_case_executions += 1
            else:
                self.counter.visible_case_executions += 1
            try:
                result = self._execute(
                    list(case.args[0]),
                    implicit_return=implicit_return,
                )
                if result == case.expected:
                    passed += 1
                elif first_error is None:
                    first_error = (
                        f"expected {case.expected!r}, got {result!r}"
                    )
            except Exception as exc:
                if first_error is None:
                    first_error = f"{type(exc).__name__}: {exc}"

        return Evaluation(True, passed, len(cases), first_error)

    def evaluate(self) -> Evaluation:
        return self._evaluate_cases(
            self.task.cases,
            hidden=False,
            implicit_return=False,
        )

    def evaluate_hidden(self) -> Evaluation:
        return self._evaluate_cases(
            self.task.hidden_cases or self.task.cases,
            hidden=True,
            implicit_return=False,
        )

    def source(self) -> str:
        lines = ["solve xs"]

        def render_body(body: list[Any], indent: int) -> None:
            prefix = "  " * indent
            for stmt in body:
                if isinstance(stmt, Init):
                    lines.append(f"{prefix}acc := {stmt.expr}")
                elif isinstance(stmt, Loop):
                    lines.append(f"{prefix}each x in xs")
                    render_body(stmt.body, indent + 1)
                    lines.append(f"{prefix}end")
                elif isinstance(stmt, Update):
                    if stmt.op == "SETX_DIRECT":
                        text = "acc := x"
                    elif stmt.op == "INC1":
                        text = "acc += 1"
                    elif stmt.op == "ADDX":
                        text = "acc += x"
                    else:
                        text = stmt.op
                    lines.append(prefix + text)
                elif isinstance(stmt, When):
                    update = {
                        "SETX": "acc := x",
                        "INC1": "acc += 1",
                        "ADDX": "acc += x",
                    }[stmt.op]
                    lines.append(
                        f"{prefix}when {stmt.condition} => {update}"
                    )
                elif isinstance(stmt, Return):
                    lines.append(f"{prefix}yield {stmt.expr}")

        render_body(self.root, 1)
        lines.append("end")
        return "\n".join(lines) + "\n"
