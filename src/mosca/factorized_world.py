from __future__ import annotations

from .native_world import Init, Loop, NativeFrame, NativeWorld, Return, Update, When


class FactorizedWorld(NativeWorld):
    """Factor the loop choice into effect x condition.

    There are three update effects and seven application scopes:
      3 x 7 = 21
    which matches the original loop's semantic alternatives while reducing
    peak local branching from 21 to 7.
    """

    EFFECTS = ("SETX", "ADDX", "INC1")
    SCOPES = ("ALWAYS", ">0", "<0", "==0", ">acc", "<acc", "==acc")

    def reset(self) -> dict:
        self.pending_effect: str | None = None
        return super().reset()

    def valid_actions(self) -> tuple[str, ...]:
        if self.done:
            return ()

        if self.pending_effect is not None:
            return tuple(f"APPLY:{scope}" for scope in self.SCOPES)

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
            if self.max_steps - len(self.actions) >= 2:
                actions.extend(f"OP:{effect}" for effect in self.EFFECTS)
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
        phase = "scope" if self.pending_effect is not None else "normal"
        observation["first_hole"] = f"factorized:{phase}"
        observation["tags"].update({
            "phase:normal": int(phase == "normal"),
            "phase:scope": int(phase == "scope"),
        })
        observation["valid_actions"] = (
            list(self.valid_actions()) if not self.done else []
        )
        return observation

    def step(self, action: str):
        if action not in self.valid_actions():
            raise ValueError(f"invalid factorized action {action}")

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
        elif action.startswith("OP:"):
            self.pending_effect = action.split(":", 1)[1]
        elif action.startswith("APPLY:"):
            assert self.pending_effect is not None
            scope = action.split(":", 1)[1]
            effect = self.pending_effect
            self.pending_effect = None
            if scope == "ALWAYS":
                self._append(
                    Update(
                        {
                            "SETX": "SETX_DIRECT",
                            "ADDX": "ADDX",
                            "INC1": "INC1",
                        }[effect]
                    )
                )
            else:
                self._append(When(f"x{scope}", effect))
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
