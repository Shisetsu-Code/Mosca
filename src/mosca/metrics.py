from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class WorkMetrics:
    environments_created: int = 0
    state_transitions: int = 0
    compile_calls: int = 0
    probe_evaluations: int = 0
    visible_evaluations: int = 0
    hidden_evaluations: int = 0
    visible_case_executions: int = 0
    hidden_case_executions: int = 0

    def as_dict(self) -> dict[str, int]:
        return asdict(self)
