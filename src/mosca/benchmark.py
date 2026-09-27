from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass

from .ast_env import ASTMaze
from .fly import FlyConfig, SparseFlyAgent
from .tasks import TASKS, Task


@dataclass
class RunStats:
    episodes: int
    solved: int
    first_solved: int | None
    best_score: float
    mean_terminal_score: float
    parameters: int = 0


def _episode_random(task: Task, rng: random.Random, max_steps: int) -> float:
    env = ASTMaze(task, max_steps=max_steps)
    while not env.done:
        actions = env.valid_actions()
        if not actions:
            break
        env.step(rng.choice(actions))
    return env.last_evaluation.score if env.last_evaluation else 0.0


def random_baseline(task: Task, episodes: int = 200, seed: int = 0, max_steps: int = 36) -> RunStats:
    rng = random.Random(seed)
    scores: list[float] = []
    first: int | None = None
    solved = 0
    for episode in range(1, episodes + 1):
        score = _episode_random(task, rng, max_steps)
        scores.append(score)
        if score == 1.0:
            solved += 1
            first = first or episode
    return RunStats(episodes, solved, first, max(scores, default=0.0), sum(scores) / max(1, len(scores)))


def train_fly(task: Task, episodes: int = 200, seed: int = 0, max_steps: int = 36) -> tuple[RunStats, SparseFlyAgent]:
    agent = SparseFlyAgent(FlyConfig(), seed=seed)
    scores: list[float] = []
    solved = 0
    first: int | None = None

    for episode in range(1, episodes + 1):
        env = ASTMaze(task, max_steps=max_steps)
        agent.begin_episode()
        observation = env.observe()
        while not env.done:
            valid = env.valid_actions()
            if not valid:
                break
            action, features = agent.choose(observation, valid, explore=True)
            next_observation, reward, done, _ = env.step(action)
            agent.learn(features, action, reward, next_observation, env.valid_actions(), done)
            observation = next_observation
        score = env.last_evaluation.score if env.last_evaluation else 0.0
        scores.append(score)
        if score == 1.0:
            solved += 1
            first = first or episode

    return RunStats(
        episodes, solved, first, max(scores, default=0.0),
        sum(scores) / max(1, len(scores)), agent.parameter_count(),
    ), agent


def benchmark(episodes: int = 200, seed: int = 0) -> dict:
    result = {"episodes": episodes, "seed": seed, "tasks": {}}
    for offset, (name, task) in enumerate(TASKS.items()):
        random_stats = random_baseline(task, episodes=episodes, seed=seed + offset)
        fly_stats, _ = train_fly(task, episodes=episodes, seed=seed + offset)
        result["tasks"][name] = {"random": asdict(random_stats), "fly": asdict(fly_stats)}
    return result


def benchmark_json(episodes: int = 200, seed: int = 0) -> str:
    return json.dumps(benchmark(episodes, seed), indent=2, sort_keys=True)


def _episode_random_motor(task: Task, rng: random.Random, max_steps: int = 8):
    from .motor_env import MotorMaze

    env = MotorMaze(task, max_steps=max_steps)
    while not env.done:
        actions = env.valid_actions()
        if not actions:
            break
        env.step(rng.choice(actions))
    score = env.last_evaluation.score if env.last_evaluation else 0.0
    return score, env


def random_motor_baseline(task: Task, episodes: int = 300, seed: int = 0, max_steps: int = 8) -> dict:
    rng = random.Random(seed)
    solved = 0
    first = None
    best = 0.0
    scores: list[float] = []
    shortest_steps = None
    shortest_source = None
    for episode in range(1, episodes + 1):
        score, env = _episode_random_motor(task, rng, max_steps)
        scores.append(score)
        best = max(best, score)
        if score == 1.0:
            solved += 1
            first = first or episode
            steps = len(env.actions)
            if shortest_steps is None or steps < shortest_steps:
                shortest_steps = steps
                shortest_source = env.source()
    return {
        "episodes": episodes,
        "solved": solved,
        "first_solved": first,
        "best_score": best,
        "mean_terminal_score": sum(scores) / max(1, len(scores)),
        "shortest_solution_steps": shortest_steps,
        "shortest_solution_source": shortest_source,
    }


def train_motor_fly(task: Task, episodes: int = 300, seed: int = 0, max_steps: int = 8) -> dict:
    from .motor_env import MotorMaze

    config = FlyConfig(
        epsilon=0.30,
        alpha=0.065,
        gamma=0.98,
        trace_decay=0.93,
        history=8,
    )
    agent = SparseFlyAgent(config, seed=seed)
    solved = 0
    first = None
    best = 0.0
    scores: list[float] = []
    shortest_steps = None
    shortest_source = None

    for episode in range(1, episodes + 1):
        env = MotorMaze(task, max_steps=max_steps)
        agent.begin_episode()
        observation = env.observe()
        while not env.done:
            actions = env.valid_actions()
            if not actions:
                break
            action, features = agent.choose(observation, actions, explore=True)
            next_observation, reward, done, _ = env.step(action)
            agent.learn(features, action, reward, next_observation, env.valid_actions(), done)
            observation = next_observation

        score = env.last_evaluation.score if env.last_evaluation else 0.0
        scores.append(score)
        best = max(best, score)
        if score == 1.0:
            solved += 1
            first = first or episode
            steps = len(env.actions)
            if shortest_steps is None or steps < shortest_steps:
                shortest_steps = steps
                shortest_source = env.source()

    return {
        "episodes": episodes,
        "solved": solved,
        "first_solved": first,
        "best_score": best,
        "mean_terminal_score": sum(scores) / max(1, len(scores)),
        "parameters": agent.parameter_count(),
        "shortest_solution_steps": shortest_steps,
        "shortest_solution_source": shortest_source,
    }


def motor_benchmark(episodes: int = 300, seed: int = 42) -> dict:
    result = {"episodes": episodes, "seed": seed, "tasks": {}}
    for offset, (name, task) in enumerate(TASKS.items()):
        task_seed = seed + offset
        result["tasks"][name] = {
            "random": random_motor_baseline(task, episodes=episodes, seed=task_seed),
            "fly": train_motor_fly(task, episodes=episodes, seed=task_seed),
        }
    return result
