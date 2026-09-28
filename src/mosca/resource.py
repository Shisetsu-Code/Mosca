from __future__ import annotations

import gc
import random
import time
import tracemalloc
from typing import Callable, TypeVar

from .benchmark import _learn_motor_episode, _motor_config
from .fly import SparseFlyAgent
from .mcts import mcts_solve
from .motor_env import MotorMaze, OracleCounter
from .tasks import TASKS, TRANSFER_TASKS, Task

T = TypeVar("T")


def _profile(fn: Callable[[], T]) -> tuple[T, dict[str, float | int]]:
    gc.collect()
    tracemalloc.start()
    wall0 = time.perf_counter()
    cpu0 = time.process_time()
    result = fn()
    cpu = time.process_time() - cpu0
    wall = time.perf_counter() - wall0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, {
        "wall_seconds": wall,
        "cpu_seconds": cpu,
        "peak_tracemalloc_bytes": peak,
    }


def _fly_until_generalized(
    agent: SparseFlyAgent,
    task: Task,
    episodes: int,
    counter: OracleCounter,
) -> dict:
    for episode in range(1, episodes + 1):
        _, hidden = _learn_motor_episode(agent, task, counter=counter)
        if hidden == 1.0:
            return {
                "success": True,
                "episodes_used": episode,
                "oracle": counter.snapshot(),
            }
    return {
        "success": False,
        "episodes_used": episodes,
        "oracle": counter.snapshot(),
    }


def _transfer_pipeline(
    seed: int,
    pretrain_episodes: int,
    adapt_episodes: int,
) -> dict:
    sources = tuple(TASKS.values())
    target = TRANSFER_TASKS["sum_positive"]
    agent = SparseFlyAgent(_motor_config(), seed=seed)

    pretrain_counter = OracleCounter()
    wall0 = time.perf_counter()
    cpu0 = time.process_time()
    for episode in range(pretrain_episodes):
        _learn_motor_episode(
            agent,
            sources[episode % len(sources)],
            counter=pretrain_counter,
        )
    pretrain_wall = time.perf_counter() - wall0
    pretrain_cpu = time.process_time() - cpu0

    adapt_counter = OracleCounter()
    wall0 = time.perf_counter()
    cpu0 = time.process_time()
    adaptation = _fly_until_generalized(
        agent, target, adapt_episodes, adapt_counter
    )
    adapt_wall = time.perf_counter() - wall0
    adapt_cpu = time.process_time() - cpu0

    return {
        "success": adaptation["success"],
        "pretrain_episodes": pretrain_episodes,
        "adapt_episodes_used": adaptation["episodes_used"],
        "pretrain_oracle": pretrain_counter.snapshot(),
        "adapt_oracle": adaptation["oracle"],
        "pretrain_wall_seconds": pretrain_wall,
        "pretrain_cpu_seconds": pretrain_cpu,
        "adapt_wall_seconds": adapt_wall,
        "adapt_cpu_seconds": adapt_cpu,
        "parameters": agent.parameter_count(),
    }


def _scratch_pipeline(seed: int, episodes: int) -> dict:
    target = TRANSFER_TASKS["sum_positive"]
    agent = SparseFlyAgent(_motor_config(), seed=seed)
    counter = OracleCounter()
    result = _fly_until_generalized(agent, target, episodes, counter)
    result["parameters"] = agent.parameter_count()
    return result


def _random_until_generalized(
    task: Task,
    episodes: int,
    seed: int,
) -> dict:
    rng = random.Random(seed)
    counter = OracleCounter()
    for episode in range(1, episodes + 1):
        env = MotorMaze(task, max_steps=8, counter=counter)
        while not env.done:
            actions = env.valid_actions()
            if not actions:
                break
            env.step(rng.choice(actions))
        if env.last_evaluation is not None and env.last_evaluation.score == 1.0:
            if env.evaluate_hidden().score == 1.0:
                return {
                    "success": True,
                    "episodes_used": episode,
                    "oracle": counter.snapshot(),
                }
    return {
        "success": False,
        "episodes_used": episodes,
        "oracle": counter.snapshot(),
    }


def resource_benchmark(
    seed: int = 0,
    pretrain_episodes: int = 600,
    adapt_episodes: int = 250,
    mcts_simulations: int = 500,
    random_episodes: int = 500,
) -> dict:
    target = TRANSFER_TASKS["sum_positive"]

    transfer, transfer_resources = _profile(
        lambda: _transfer_pipeline(seed, pretrain_episodes, adapt_episodes)
    )
    scratch, scratch_resources = _profile(
        lambda: _scratch_pipeline(seed, adapt_episodes)
    )
    mcts, mcts_resources = _profile(
        lambda: mcts_solve(
            target,
            simulations=mcts_simulations,
            seed=seed,
            stop_on_generalizing=True,
        )
    )
    random_result, random_resources = _profile(
        lambda: _random_until_generalized(target, random_episodes, seed)
    )

    return {
        "seed": seed,
        "target": target.name,
        "budgets": {
            "pretrain_episodes": pretrain_episodes,
            "adapt_episodes": adapt_episodes,
            "mcts_simulations": mcts_simulations,
            "random_episodes": random_episodes,
        },
        "transfer": {
            **transfer,
            "resources": transfer_resources,
        },
        "scratch": {
            **scratch,
            "resources": scratch_resources,
        },
        "mcts": {
            "success": mcts.first_generalizing_at is not None,
            "simulations_used": mcts.simulations,
            "oracle": {
                "visible_oracle_calls": mcts.visible_oracle_calls,
                "visible_case_executions": mcts.visible_case_executions,
                "hidden_oracle_calls": mcts.hidden_oracle_calls,
                "hidden_case_executions": mcts.hidden_case_executions,
            },
            "resources": mcts_resources,
        },
        "random": {
            **random_result,
            "resources": random_resources,
        },
    }
