from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from .env import Action, PythonMaze
from .tasks import Task


@dataclass(frozen=True)
class SearchResult:
    actions: tuple[Action, ...]
    source: str
    explored: int
    score: float


def replay(task: Task, actions: tuple[Action, ...], max_steps: int = 20) -> PythonMaze:
    env = PythonMaze(task, max_steps=max_steps)
    for action in actions:
        env.step(action)
        if env.done:
            break
    return env


def bfs_solve(task: Task, max_depth: int = 10) -> SearchResult | None:
    queue = deque([tuple()])
    seen: set[tuple[str, tuple[str, ...]]] = set()
    explored = 0

    while queue:
        actions = queue.popleft()
        env = replay(task, actions, max_steps=max_depth + 1)
        explored += 1

        if env.done:
            evaluation = env.last_evaluation
            if evaluation and evaluation.score == 1.0:
                return SearchResult(actions, env.source(), explored, evaluation.score)
            continue

        key = (env.source(), tuple(b.kind for b in env.blocks))
        if key in seen:
            continue
        seen.add(key)

        if len(actions) >= max_depth:
            continue

        for action in env.valid_actions():
            queue.append(actions + (action,))

    return None
