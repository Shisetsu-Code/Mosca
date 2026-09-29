from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass

from .motor_env import OracleCounter
from .tasks import TASKS, TRANSFER_TASKS, Task
from .worlds import ProgramWorld, create_world


@dataclass(frozen=True)
class MCTSResult:
    task: str
    simulations: int
    first_visible_at: int | None
    first_generalizing_at: int | None
    best_train_score: float
    best_hidden_score: float
    actions: tuple[str, ...] | None
    source: str | None
    visible_oracle_calls: int
    visible_case_executions: int
    hidden_oracle_calls: int
    hidden_case_executions: int
    visible_oracle_calls_at_first_generalizing: int | None
    visible_case_executions_at_first_generalizing: int | None


def replay(
    task: Task,
    actions: tuple[str, ...],
    max_steps: int = 8,
    counter: OracleCounter | None = None,
    world_name: str = "python",
) -> ProgramWorld:
    env = create_world(
        world_name,
        task,
        max_steps=max_steps,
        counter=counter,
    )
    for action in actions:
        if env.done:
            break
        if action not in env.valid_actions():
            break
        env.step(action)
    return env


def _visible_score(env: ProgramWorld) -> float:
    if env.last_evaluation is not None:
        return env.last_evaluation.score
    return env.probe_score()


def mcts_solve(
    task: Task,
    simulations: int = 500,
    seed: int = 0,
    max_steps: int = 8,
    exploration: float = 1.4,
    stop_on_generalizing: bool = False,
    world_name: str = "python",
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
    first_visible_at: int | None = None
    first_generalizing_at: int | None = None
    first_generalizing_oracle: dict[str, int] | None = None
    counter = OracleCounter()
    executed_simulations = 0

    for simulation in range(1, simulations + 1):
        executed_simulations = simulation
        prefix: tuple[str, ...] = ()
        path = [prefix]
        env = replay(
            task,
            prefix,
            max_steps=max_steps,
            counter=counter,
            world_name=world_name,
        )

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
                env = replay(
                    task,
                    prefix,
                    max_steps=max_steps,
                    counter=counter,
                    world_name=world_name,
                )
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
            env = replay(
                task,
                prefix,
                max_steps=max_steps,
                counter=counter,
                world_name=world_name,
            )

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
            if first_visible_at is None:
                first_visible_at = simulation
            hidden_score = env.evaluate_hidden().score
            if hidden_score > best_hidden:
                best_hidden = hidden_score
                best_actions = rollout
                best_source = env.source()
            if hidden_score == 1.0 and first_generalizing_at is None:
                first_generalizing_at = simulation
                first_generalizing_oracle = counter.snapshot()
                best_train = 1.0
                best_hidden = 1.0
                best_actions = rollout
                best_source = env.source()

        # Hidden score is intentionally excluded from the backed-up reward.
        reward = train_score
        for node in path:
            visits[node] = visits.get(node, 0) + 1
            values[node] = values.get(node, 0.0) + reward

        if stop_on_generalizing and first_generalizing_at is not None:
            break

    return MCTSResult(
        task=task.name,
        simulations=executed_simulations,
        first_visible_at=first_visible_at,
        first_generalizing_at=first_generalizing_at,
        best_train_score=best_train,
        best_hidden_score=best_hidden,
        actions=best_actions,
        source=best_source,
        visible_oracle_calls=counter.visible_oracle_calls,
        visible_case_executions=counter.visible_case_executions,
        hidden_oracle_calls=counter.hidden_oracle_calls,
        hidden_case_executions=counter.hidden_case_executions,
        visible_oracle_calls_at_first_generalizing=(
            first_generalizing_oracle["visible_oracle_calls"]
            if first_generalizing_oracle else None
        ),
        visible_case_executions_at_first_generalizing=(
            first_generalizing_oracle["visible_case_executions"]
            if first_generalizing_oracle else None
        ),
    )


def mcts_benchmark(
    simulations: int = 500,
    seed: int = 42,
    world_name: str = "python",
) -> dict:
    tasks = {**TASKS, **TRANSFER_TASKS}
    results = {}
    for offset, (name, task) in enumerate(tasks.items()):
        result = mcts_solve(
            task,
            simulations=simulations,
            seed=seed + offset,
            world_name=world_name,
        )
        results[name] = asdict(result)
    return {
        "simulations_budget": simulations,
        "seed": seed,
        "world": world_name,
        "tasks": results,
    }



def mcts_multiseed(
    simulations: int = 500,
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
    world_name: str = "python",
) -> dict:
    tasks = {**TASKS, **TRANSFER_TASKS}
    per_task: dict[str, dict] = {}

    for task_index, (name, task) in enumerate(tasks.items()):
        results = [
            mcts_solve(
                task,
                simulations=simulations,
                seed=seed + task_index,
                world_name=world_name,
            )
            for seed in seeds
        ]
        solved = [r.first_generalizing_at for r in results if r.first_generalizing_at is not None]
        oracle_solved = [
            r for r in results if r.visible_oracle_calls_at_first_generalizing is not None
        ]
        per_task[name] = {
            "seed_successes": len(solved),
            "mean_first_generalizing_at": (sum(solved) / len(solved)) if solved else None,
            "mean_visible_oracle_calls_at_first_generalizing": (
                sum(r.visible_oracle_calls_at_first_generalizing for r in oracle_solved)
                / len(oracle_solved)
                if oracle_solved else None
            ),
            "mean_visible_case_executions_at_first_generalizing": (
                sum(r.visible_case_executions_at_first_generalizing for r in oracle_solved)
                / len(oracle_solved)
                if oracle_solved else None
            ),
            "best_first_generalizing_at": min(solved) if solved else None,
            "worst_first_generalizing_at": max(solved) if solved else None,
            "runs": [asdict(r) for r in results],
        }

    return {
        "simulations_budget": simulations,
        "seeds": list(seeds),
        "world": world_name,
        "tasks": per_task,
    }
