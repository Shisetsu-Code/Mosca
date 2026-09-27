from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass

from .motor_env import MotorMaze
from .tasks import TASKS, TRANSFER_TASKS, Task


@dataclass(frozen=True)
class MCTSResult:
    task: str
    simulations: int
    solved_at: int | None
    best_train_score: float
    hidden_score: float
    actions: tuple[str, ...] | None
    source: str | None


def replay(task: Task, actions: tuple[str, ...], max_steps: int = 8) -> MotorMaze:
    env = MotorMaze(task, max_steps=max_steps)
    for action in actions:
        if env.done:
            break
        if action not in env.valid_actions():
            break
        env.step(action)
    return env


def _visible_score(env: MotorMaze) -> float:
    if env.last_evaluation is not None:
        return env.last_evaluation.score
    return env.probe_score()


def mcts_solve(
    task: Task,
    simulations: int = 500,
    seed: int = 0,
    max_steps: int = 8,
    exploration: float = 1.4,
) -> MCTSResult:
    """UCT search over the same semantic action space used by SparseFlyAgent.

    Search and backpropagation use visible training cases only. Hidden cases are
    consulted solely after a visible solution is found.
    """

    rng = random.Random(seed)
    visits: dict[tuple[str, ...], int] = {(): 0}
    values: dict[tuple[str, ...], float] = {(): 0.0}
    untried: dict[tuple[str, ...], list[str]] = {}
    best_train = 0.0
    best_actions: tuple[str, ...] | None = None
    best_source: str | None = None
    best_hidden = 0.0
    solved_at: int | None = None

    for simulation in range(1, simulations + 1):
        prefix: tuple[str, ...] = ()
        path = [prefix]
        env = replay(task, prefix, max_steps=max_steps)

        # Selection + one-node expansion.
        while not env.done:
            valid = tuple(env.valid_actions())
            if not valid:
                break

            if prefix not in untried:
                items = list(valid)
                rng.shuffle(items)
                untried[prefix] = items

            if untried[prefix]:
                action = untried[prefix].pop()
                prefix = prefix + (action,)
                path.append(prefix)
                visits.setdefault(prefix, 0)
                values.setdefault(prefix, 0.0)
                env = replay(task, prefix, max_steps=max_steps)
                break

            parent_visits = max(1, visits.get(prefix, 0))
            candidates: list[tuple[float, float, str]] = []
            for action in valid:
                child = prefix + (action,)
                child_visits = visits.get(child, 0)
                if child_visits == 0:
                    ucb = float("inf")
                else:
                    mean = values.get(child, 0.0) / child_visits
                    ucb = mean + exploration * math.sqrt(math.log(parent_visits + 1) / child_visits)
                candidates.append((ucb, rng.random(), action))
            action = max(candidates)[2]
            prefix = prefix + (action,)
            path.append(prefix)
            visits.setdefault(prefix, 0)
            values.setdefault(prefix, 0.0)
            env = replay(task, prefix, max_steps=max_steps)

        # Random rollout from the expanded/selected state.
        rollout = prefix
        while not env.done:
            valid = tuple(env.valid_actions())
            if not valid:
                break
            action = rng.choice(valid)
            rollout = rollout + (action,)
            env.step(action)

        train_score = _visible_score(env)
        if train_score > best_train:
            best_train = train_score
            best_actions = rollout
            best_source = env.source()

        hidden_score = 0.0
        if env.last_evaluation is not None and env.last_evaluation.score == 1.0:
            hidden_score = env.evaluate_hidden().score
            if hidden_score > best_hidden:
                best_hidden = hidden_score
                best_actions = rollout
                best_source = env.source()
            if hidden_score == 1.0 and solved_at is None:
                solved_at = simulation
                best_train = 1.0
                best_hidden = 1.0
                best_actions = rollout
                best_source = env.source()

        # Hidden score is intentionally excluded from the backed-up reward.
        reward = train_score
        for node in path:
            visits[node] = visits.get(node, 0) + 1
            values[node] = values.get(node, 0.0) + reward

        if solved_at is not None:
            break

    return MCTSResult(
        task=task.name,
        simulations=simulation if simulations else 0,
        solved_at=solved_at,
        best_train_score=best_train,
        hidden_score=best_hidden,
        actions=best_actions,
        source=best_source,
    )


def mcts_benchmark(simulations: int = 500, seed: int = 42) -> dict:
    tasks = {**TASKS, **TRANSFER_TASKS}
    results = {}
    for offset, (name, task) in enumerate(tasks.items()):
        result = mcts_solve(task, simulations=simulations, seed=seed + offset)
        results[name] = asdict(result)
    return {
        "simulations_budget": simulations,
        "seed": seed,
        "tasks": results,
    }



def mcts_multiseed(
    simulations: int = 500,
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
) -> dict:
    tasks = {**TASKS, **TRANSFER_TASKS}
    per_task: dict[str, dict] = {}

    for task_index, (name, task) in enumerate(tasks.items()):
        results = [
            mcts_solve(task, simulations=simulations, seed=seed + task_index)
            for seed in seeds
        ]
        solved = [r.solved_at for r in results if r.solved_at is not None]
        per_task[name] = {
            "seed_successes": len(solved),
            "mean_solved_at": (sum(solved) / len(solved)) if solved else None,
            "best_solved_at": min(solved) if solved else None,
            "worst_solved_at": max(solved) if solved else None,
            "runs": [asdict(r) for r in results],
        }

    return {
        "simulations_budget": simulations,
        "seeds": list(seeds),
        "tasks": per_task,
    }
