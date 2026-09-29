from __future__ import annotations

from .native_world import NativeWorld


class TypedWorld(NativeWorld):
    """Role-constrained reduction language over the native backend.

    The world keeps the original one-action semantic depth but prunes
    combinations that do not match the current reduction role:
      - additive/count effects may be filtered by x versus 0;
      - candidate replacement compares x versus acc.
    """

    def valid_actions(self) -> tuple[str, ...]:
        if self.done:
            return ()

        if self.acc_defined and self.probe_score() >= 1.0 - 1e-12:
            if len(self.stack) > 1:
                return ("END",)
            return ("RETURN:acc",)

        if not self.acc_defined:
            if len(self.stack) == 1:
                return (
                    "SET:acc=-1",
                    "SET:acc=0",
                    "SET:acc=1",
                    "SET:acc=xs[0]",
                )
            return ()

        actions: list[str] = []
        if len(self.stack) == 1:
            actions.extend(("FOR:x:xs", "RETURN:acc"))
        elif self.frame.kind == "for":
            actions.extend(("SET:acc=x", "AUG:acc+=x", "AUG:acc+=1"))

            # Filtered additive/count reductions: condition depends only on x.
            for cmp in (">", "<", "=="):
                actions.extend((
                    f"WHEN:x{cmp}0:ADDX",
                    f"WHEN:x{cmp}0:INC1",
                ))

            # Candidate replacement: compare the new candidate with acc.
            for cmp in (">", "<", "=="):
                actions.append(f"WHEN:x{cmp}acc:SETX")

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

        return tuple(actions)
