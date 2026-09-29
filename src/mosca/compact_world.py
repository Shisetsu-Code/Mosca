from __future__ import annotations

from .native_world import Init, Loop, NativeFrame, NativeWorld, Return, Update, When


class CompactWorld(NativeWorld):
    """Staged low-branching language over the same native semantic IR.

    Composite conditional actions are factored into:
      GUARD -> CMP -> RHS -> DO

    This preserves the set of expressible programs while trading additional
    decision depth for a much smaller local branching factor.
    """

    def reset(self) -> dict:
        self.pending_stage: str | None = None
        self.pending_cmp: str | None = None
        self.pending_rhs: str | None = None
        return super().reset()

    def valid_actions(self) -> tuple[str, ...]:
        if self.done:
            return ()

        if self.pending_stage == "cmp":
            return ("CMP:>", "CMP:<", "CMP:==")
        if self.pending_stage == "rhs":
            return ("RHS:0", "RHS:acc")
        if self.pending_stage == "do":
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
            # GUARD requires four decisions including GUARD itself.
            if self.max_steps - len(self.actions) >= 4:
                actions.append("GUARD")
            if self.frame.body:
                actions.append("END")

        remaining = self.max_steps - len(self.actions)
        if remaining <= 2:
            closers = [a for a in actions if a in {"END", "YIELD"}]
            if closers:
                actions = closers
        return tuple(actions)

    def observe(self) -> dict:
        observation = super().observe()
        phase = self.pending_stage or "normal"
        observation["first_hole"] = f"compact:{phase}"
        observation["tags"].update({
            "phase:normal": int(phase == "normal"),
            "phase:cmp": int(phase == "cmp"),
            "phase:rhs": int(phase == "rhs"),
            "phase:do": int(phase == "do"),
        })
        observation["valid_actions"] = (
            list(self.valid_actions()) if not self.done else []
        )
        return observation

    def _clear_pending(self) -> None:
        self.pending_stage = None
        self.pending_cmp = None
        self.pending_rhs = None

    def step(self, action: str):
        if action not in self.valid_actions():
            raise ValueError(f"invalid compact action {action}")

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
        elif action == "GUARD":
            self.pending_stage = "cmp"
        elif action.startswith("CMP:"):
            self.pending_cmp = action.split(":", 1)[1]
            self.pending_stage = "rhs"
        elif action.startswith("RHS:"):
            self.pending_rhs = action.split(":", 1)[1]
            self.pending_stage = "do"
        elif action.startswith("DO:"):
            op = action.split(":", 1)[1]
            assert self.pending_cmp is not None
            assert self.pending_rhs is not None
            self._append(
                When(
                    f"x{self.pending_cmp}{self.pending_rhs}",
                    op,
                )
            )
            self._clear_pending()
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
