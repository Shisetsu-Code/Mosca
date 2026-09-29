from __future__ import annotations

from .native_world import Init, Loop, NativeFrame, NativeWorld, Return, Update, When


class Compact2World(NativeWorld):
    """Moderately compact language: condition in one action, update in another."""

    CONDITIONS = (">0", "<0", "==0", ">acc", "<acc", "==acc")

    def reset(self) -> dict:
        self.pending_condition: str | None = None
        return super().reset()

    def valid_actions(self) -> tuple[str, ...]:
        if self.done:
            return ()

        if self.pending_condition is not None:
            return ("DO:SETX", "DO:INC1", "DO:ADDX")

        if self.acc_defined and self.probe_score() >= 1.0 - 1e-12:
            if len(self.stack) > 1:
                return ("END",)
            return ("YIELD",)

        if not self.acc_defined:
            if len(self.stack) == 1:
                return ("INIT:-1", "INIT:0", "INIT:1", "INIT:first")
            return ()

        actions: list[str] = []
        if len(self.stack) == 1:
            actions.extend(("EACH", "YIELD"))
        elif self.frame.kind == "for":
            actions.extend(("SETX", "ADDX", "INC1"))
            if self.max_steps - len(self.actions) >= 2:
                actions.extend(f"COND:{cond}" for cond in self.CONDITIONS)
            if self.frame.body:
                actions.append("END")

        remaining = self.max_steps - len(self.actions)
        if remaining <= 1:
            closers = [a for a in actions if a in {"END", "YIELD"}]
            if closers:
                actions = closers
        return tuple(actions)

    def observe(self) -> dict:
        observation = super().observe()
        phase = "do" if self.pending_condition is not None else "normal"
        observation["first_hole"] = f"compact2:{phase}"
        observation["tags"].update({
            "phase:normal": int(phase == "normal"),
            "phase:do": int(phase == "do"),
        })
        observation["valid_actions"] = (
            list(self.valid_actions()) if not self.done else []
        )
        return observation

    def step(self, action: str):
        if action not in self.valid_actions():
            raise ValueError(f"invalid compact2 action {action}")

        before_probe = self.probe_score()
        self.actions.append(action)
        reward = -0.02
        info: dict = {}

        if action.startswith("INIT:"):
            rhs = action.split(":", 1)[1]
            expr = "xs[0]" if rhs == "first" else rhs
            self._append(Init(expr))
            self.acc_defined = True
        elif action == "EACH":
            loop = Loop([])
            self._append(loop)
            self.stack.append(NativeFrame("for", loop.body))
        elif action == "SETX":
            self._append(Update("SETX_DIRECT"))
        elif action == "ADDX":
            self._append(Update("ADDX"))
        elif action == "INC1":
            self._append(Update("INC1"))
        elif action.startswith("COND:"):
            self.pending_condition = action.split(":", 1)[1]
        elif action.startswith("DO:"):
            assert self.pending_condition is not None
            op = action.split(":", 1)[1]
            self._append(When(f"x{self.pending_condition}", op))
            self.pending_condition = None
        elif action == "END":
            self.stack.pop()
        elif action == "YIELD":
            self._append(Return("acc"))
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
            reward += 0.30 if action in {"END", "YIELD"} else -0.30
        elif after_probe >= 1.0 - 1e-12:
            reward += 0.50

        info["probe_before"] = before_probe
        info["probe_after"] = after_probe

        if len(self.actions) >= self.max_steps and not self.done:
            self.done = True
            reward -= 0.5

        return self.observe(), reward, self.done, info
