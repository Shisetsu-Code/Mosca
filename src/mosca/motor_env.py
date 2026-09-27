from __future__ import annotations

import ast
import copy
import warnings
from dataclasses import dataclass

from .sensory import task_sensory_tokens\nfrom .tasks import Task


@dataclass(frozen=True)
class Evaluation:
    compiled: bool
    passed: int
    total: int
    error: str | None = None

    @property
    def score(self) -> float:
        return self.passed / self.total if self.total else 0.0


@dataclass
class Frame:
    kind: str
    body: list[ast.stmt]


class MotorMaze:
    """Hierarchical semantic action layer over CPython AST.

    The learner chooses statement-sized motor primitives generated from variables,
    constants and operators currently in scope. It never emits Python text.
    """

    CMPS = (">", "<", "==")
    CONSTS = (-1, 0, 1)

    def __init__(self, task: Task, max_steps: int = 12):
        self.task = task
        self.max_steps = max_steps
        self.sensory = task_sensory_tokens(task)
        self.reset()

    def reset(self) -> dict:
        self.root: list[ast.stmt] = []
        self.stack: list[Frame] = [Frame("function", self.root)]
        self.acc_defined = False
        self.actions: list[str] = []
        self.done = False
        self.last_evaluation: Evaluation | None = None
        self._probe_cache: float | None = None
        return self.observe()

    @property
    def frame(self) -> Frame:
        return self.stack[-1]

    @property
    def in_loop(self) -> bool:
        return any(f.kind == "for" for f in self.stack)

    @property
    def in_if(self) -> bool:
        return any(f.kind == "if" for f in self.stack)

    def conditions(self) -> list[str]:
        if not self.in_loop:
            return []
        rights = ["0"]
        if self.acc_defined:
            rights.append("acc")
        return [f"x{op}{right}" for right in rights for op in self.CMPS if right != "x"]

    def valid_actions(self) -> tuple[str, ...]:
        if self.done:
            return ()

        # Goal inhibition: when the provisional behavior already passes all
        # training tests, freeze semantics and only close/return.
        if self.acc_defined and self.probe_score() >= 1.0 - 1e-12:
            if len(self.stack) > 1:
                return ("END",)
            return ("RETURN:acc",)

        actions: list[str] = []
        # Curriculum v0: list reductions with one accumulator. These encode
        # scope/data roles, not a target program.
        if not self.acc_defined:
            if len(self.stack) == 1:
                return tuple(f"SET:acc={expr}" for expr in ("-1", "0", "1", "xs[0]"))
            return ()

        if len(self.stack) == 1:
            actions.extend(("FOR:x:xs", "RETURN:acc"))
        elif self.frame.kind == "for":
            actions.extend(("SET:acc=x", "AUG:acc+=x", "AUG:acc+=1"))
            # Generated motor options: condition x update. No task-specific
            # max/count/sum macro exists.
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
            closers = [a for a in actions if a == "END" or a.startswith("RETURN:")]
            if closers:
                actions = closers
        return tuple(dict.fromkeys(actions))

    def observe(self) -> dict:
        return {
            "first_hole": "motor",
            "sensory": self.sensory,
            "holes": max(0, len(self.stack) - 1),
            "nodes": sum(len(f.body) for f in self.stack),
            "tags": {
                "frame:function": int(len(self.stack) == 1),
                "frame:for": int(self.in_loop),
                "frame:if": int(self.in_if),
                "acc": int(self.acc_defined),
                "depth": len(self.stack) - 1,
            },
            "steps": len(self.actions),
            "valid_actions": list(self.valid_actions()) if not self.done else [],
        }

    @staticmethod
    def _expr(text: str) -> ast.expr:
        if text in {"-1", "0", "1"}:
            return ast.Constant(int(text))
        if text in {"acc", "x", "xs"}:
            return ast.Name(text, ast.Load())
        if text == "xs[0]":
            return ast.Subscript(ast.Name("xs", ast.Load()), ast.Constant(0), ast.Load())
        raise ValueError(text)

    @classmethod
    def _condition(cls, text: str) -> ast.expr:
        for symbol, op_cls in (("==", ast.Eq), (">", ast.Gt), ("<", ast.Lt)):
            if symbol in text:
                left, right = text.split(symbol, 1)
                return ast.Compare(cls._expr(left), [op_cls()], [cls._expr(right)])
        raise ValueError(text)

    def _append(self, stmt: ast.stmt) -> None:
        self.frame.body.append(stmt)

    def probe_score(self) -> float:
        if not self.acc_defined:
            return 0.0
        if self._probe_cache is not None:
            return self._probe_cache
        root = copy.deepcopy(self.root)

        def fill_empty(body: list[ast.stmt]) -> None:
            if not body:
                body.append(ast.Pass())
                return
            for stmt in body:
                if isinstance(stmt, (ast.For, ast.If)):
                    fill_empty(stmt.body)

        fill_empty(root)
        root.append(ast.Return(ast.Name("acc", ast.Load())))
        self._probe_cache = self._evaluate_root(root).score
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
            self._append(ast.Assign([ast.Name("acc", ast.Store())], self._expr(expr)))
            self.acc_defined = True
        elif action.startswith("AUG:acc"):
            rest = action[len("AUG:acc"):]
            symbol = rest[0]
            expr = rest[2:]
            op = {"+": ast.Add(), "-": ast.Sub(), "*": ast.Mult()}[symbol]
            self._append(ast.AugAssign(ast.Name("acc", ast.Store()), op, self._expr(expr)))
        elif action == "FOR:x:xs":
            node = ast.For(ast.Name("x", ast.Store()), ast.Name("xs", ast.Load()), [], [], None)
            self._append(node)
            self.stack.append(Frame("for", node.body))
        elif action.startswith("WHEN:"):
            payload = action[len("WHEN:"):]
            cond, op = payload.rsplit(":", 1)
            if op == "SETX":
                update = ast.Assign([ast.Name("acc", ast.Store())], ast.Name("x", ast.Load()))
            elif op == "INC1":
                update = ast.AugAssign(ast.Name("acc", ast.Store()), ast.Add(), ast.Constant(1))
            elif op == "ADDX":
                update = ast.AugAssign(ast.Name("acc", ast.Store()), ast.Add(), ast.Name("x", ast.Load()))
            else:
                raise ValueError(op)
            self._append(ast.If(self._condition(cond), [update], []))
        elif action.startswith("IF:"):
            node = ast.If(self._condition(action[3:]), [], [])
            self._append(node)
            self.stack.append(Frame("if", node.body))
        elif action == "END":
            self.stack.pop()
        elif action.startswith("RETURN:"):
            self._append(ast.Return(self._expr(action.split(":", 1)[1])))
            self.last_evaluation = self.evaluate()
            self.done = True
            reward += self.last_evaluation.score * 5.0
            if self.last_evaluation.score == 1.0:
                reward += 5.0
            info["evaluation"] = self.last_evaluation

        self._probe_cache = None
        after_probe = self.last_evaluation.score if self.done and self.last_evaluation else self.probe_score()
        reward += 2.0 * (after_probe - before_probe)
        if before_probe >= 1.0 - 1e-12:
            reward += 0.30 if (action == "END" or action.startswith("RETURN:")) else -0.30
        elif after_probe >= 1.0 - 1e-12:
            reward += 0.50
        info["probe_before"] = before_probe
        info["probe_after"] = after_probe

        if len(self.actions) >= self.max_steps and not self.done:
            self.done = True
            reward -= 0.5
        return self.observe(), reward, self.done, info

    def module(self) -> ast.Module:
        fn = ast.FunctionDef(
            "solve",
            ast.arguments(
                posonlyargs=[], args=[ast.arg("xs")], vararg=None,
                kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[],
            ),
            self.root or [ast.Pass()],
            [],
            None,
            None,
        )
        return ast.fix_missing_locations(ast.Module([fn], []))

    def source(self) -> str:
        return ast.unparse(self.module()) + "\n"

    def _evaluate_root(self, root: list[ast.stmt], cases=None) -> Evaluation:
        cases = self.task.cases if cases is None else cases
        fn = ast.FunctionDef(
            "solve",
            ast.arguments(
                posonlyargs=[], args=[ast.arg("xs")], vararg=None,
                kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[],
            ),
            root or [ast.Pass()], [], None, None,
        )
        module = ast.fix_missing_locations(ast.Module([fn], []))
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                code = compile(module, "<mosca-motor>", "exec", dont_inherit=True, optimize=0)
            ns = {"__builtins__": {}}
            exec(code, ns, ns)
            solve = ns["solve"]
        except Exception as exc:
            return Evaluation(False, 0, len(cases), f"{type(exc).__name__}: {exc}")

        passed = 0
        error = None
        for case in cases:
            try:
                got = solve(*case.args)
                if got == case.expected:
                    passed += 1
                elif error is None:
                    error = f"expected {case.expected!r}, got {got!r}"
            except Exception as exc:
                if error is None:
                    error = f"{type(exc).__name__}: {exc}"
        return Evaluation(True, passed, len(cases), error)

    def evaluate(self) -> Evaluation:
        return self._evaluate_root(self.root)

    def evaluate_hidden(self) -> Evaluation:
        cases = self.task.hidden_cases or self.task.cases
        return self._evaluate_root(self.root, cases)
