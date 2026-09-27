from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass

from .ast_env import ASTMaze
from .fly import FlyConfig, SparseFlyAgent
from .tasks import TASKS, TRANSFER_TASKS, Task


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
    generalized = 0
    first_generalized = None
    shortest_hidden_score = None
    for episode in range(1, episodes + 1):
        score, env = _episode_random_motor(task, rng, max_steps)
        scores.append(score)
        best = max(best, score)
        if score == 1.0:
            solved += 1
            first = first or episode
            steps = len(env.actions)
            hidden = env.evaluate_hidden().score
            if hidden == 1.0:
                generalized += 1
                first_generalized = first_generalized or episode
            if shortest_steps is None or steps < shortest_steps:
                shortest_steps = steps
                shortest_source = env.source()
                shortest_hidden_score = hidden
    return {
        "episodes": episodes,
        "solved": solved,
        "first_solved": first,
        "best_score": best,
        "mean_terminal_score": sum(scores) / max(1, len(scores)),
        "shortest_solution_steps": shortest_steps,
        "shortest_solution_source": shortest_source,
        "shortest_solution_hidden_score": shortest_hidden_score,
        "generalized": generalized,
        "first_generalized": first_generalized,
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
    generalized = 0
    first_generalized = None
    shortest_hidden_score = None

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
            hidden = env.evaluate_hidden().score
            if hidden == 1.0:
                generalized += 1
                first_generalized = first_generalized or episode
            if shortest_steps is None or steps < shortest_steps:
                shortest_steps = steps
                shortest_source = env.source()
                shortest_hidden_score = hidden

    return {
        "episodes": episodes,
        "solved": solved,
        "first_solved": first,
        "best_score": best,
        "mean_terminal_score": sum(scores) / max(1, len(scores)),
        "parameters": agent.parameter_count(),
        "shortest_solution_steps": shortest_steps,
        "shortest_solution_source": shortest_source,
        "shortest_solution_hidden_score": shortest_hidden_score,
        "generalized": generalized,
        "first_generalized": first_generalized,
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



def _motor_config() -> FlyConfig:
    return FlyConfig(
        epsilon=0.30,
        alpha=0.065,
        gamma=0.98,
        trace_decay=0.93,
        history=8,
        consolidation_rate=0.12,
        slow_mix=0.45,
        consolidation_threshold=7.0,
    )


def _learn_motor_episode(agent: SparseFlyAgent, task: Task, max_steps: int = 8) -> tuple[float, float]:
    from .motor_env import MotorMaze

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
    train_score = env.last_evaluation.score if env.last_evaluation else 0.0
    hidden_score = env.evaluate_hidden().score if train_score == 1.0 else 0.0
    return train_score, hidden_score


def motor_multiseed(episodes: int = 200, seeds: tuple[int, ...] = (0, 1, 2)) -> dict:
    runs = []
    for seed in seeds:
        result = motor_benchmark(episodes=episodes, seed=seed)
        runs.append(result)

    summary: dict[str, dict] = {}
    for name in TASKS:
        fly_successes = sum(run["tasks"][name]["fly"]["generalized"] > 0 for run in runs)
        random_successes = sum(run["tasks"][name]["random"]["generalized"] > 0 for run in runs)
        fly_first = [
            run["tasks"][name]["fly"]["first_generalized"]
            for run in runs
            if run["tasks"][name]["fly"]["first_generalized"] is not None
        ]
        random_first = [
            run["tasks"][name]["random"]["first_generalized"]
            for run in runs
            if run["tasks"][name]["random"]["first_generalized"] is not None
        ]
        summary[name] = {
            "fly_seed_successes": fly_successes,
            "random_seed_successes": random_successes,
            "fly_mean_first_generalized": (sum(fly_first) / len(fly_first)) if fly_first else None,
            "random_mean_first_generalized": (sum(random_first) / len(random_first)) if random_first else None,
        }
    return {"episodes": episodes, "seeds": list(seeds), "summary": summary, "runs": runs}


def _adapt_summary(agent: SparseFlyAgent, task: Task, episodes: int) -> dict:
    first_generalized = None
    generalized = 0
    solved = 0
    for episode in range(1, episodes + 1):
        train_score, hidden_score = _learn_motor_episode(agent, task)
        if train_score == 1.0:
            solved += 1
        if hidden_score == 1.0:
            generalized += 1
            first_generalized = first_generalized or episode
    return {
        "episodes": episodes,
        "solved": solved,
        "generalized": generalized,
        "first_generalized": first_generalized,
        "parameters": agent.parameter_count(),
    }


def transfer_benchmark(
    pretrain_episodes: int = 300,
    adapt_episodes: int = 200,
    seed: int = 42,
) -> dict:
    """Pretrain on core tasks, then adapt to a compositional unseen task."""

    sources = tuple(TASKS.values())
    target = TRANSFER_TASKS["sum_positive"]

    pretrained = SparseFlyAgent(_motor_config(), seed=seed)
    for episode in range(pretrain_episodes):
        _learn_motor_episode(pretrained, sources[episode % len(sources)])

    scratch = SparseFlyAgent(_motor_config(), seed=seed)
    transfer_zero_shot = _zero_shot_summary(pretrained, target)
    scratch_zero_shot = _zero_shot_summary(scratch, target)
    transfer = _adapt_summary(pretrained, target, adapt_episodes)
    baseline = _adapt_summary(scratch, target, adapt_episodes)
    return {
        "seed": seed,
        "pretrain_episodes": pretrain_episodes,
        "adapt_episodes": adapt_episodes,
        "target": target.name,
        "transfer_zero_shot": transfer_zero_shot,
        "scratch_zero_shot": scratch_zero_shot,
        "transfer": transfer,
        "scratch": baseline,
    }



def transfer_multiseed(
    pretrain_episodes: int = 600,
    adapt_episodes: int = 250,
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
) -> dict:
    runs = [
        transfer_benchmark(pretrain_episodes, adapt_episodes, seed)
        for seed in seeds
    ]

    def aggregate(key: str) -> dict:
        first = [
            run[key]["first_generalized"]
            for run in runs
            if run[key]["first_generalized"] is not None
        ]
        return {
            "seed_successes": sum(run[key]["generalized"] > 0 for run in runs),
            "total_generalized": sum(run[key]["generalized"] for run in runs),
            "mean_first_generalized": (sum(first) / len(first)) if first else None,
            "best_first_generalized": min(first) if first else None,
        }

    transfer = aggregate("transfer")
    scratch = aggregate("scratch")
    zero_shot = {
        "transfer_seed_successes": sum(run["transfer_zero_shot"]["generalized"] > 0 for run in runs),
        "scratch_seed_successes": sum(run["scratch_zero_shot"]["generalized"] > 0 for run in runs),
        "transfer_total_generalized": sum(run["transfer_zero_shot"]["generalized"] for run in runs),
        "scratch_total_generalized": sum(run["scratch_zero_shot"]["generalized"] for run in runs),
    }
    paired_wins = 0
    paired_losses = 0
    paired_ties = 0
    for run in runs:
        a = run["transfer"]["first_generalized"]
        b = run["scratch"]["first_generalized"]
        if a is None and b is None:
            paired_ties += 1
        elif a is None:
            paired_losses += 1
        elif b is None:
            paired_wins += 1
        elif a < b:
            paired_wins += 1
        elif a > b:
            paired_losses += 1
        else:
            paired_ties += 1

    return {
        "pretrain_episodes": pretrain_episodes,
        "adapt_episodes": adapt_episodes,
        "seeds": list(seeds),
        "transfer": transfer,
        "scratch": scratch,
        "zero_shot": zero_shot,
        "paired": {
            "transfer_wins": paired_wins,
            "scratch_wins": paired_losses,
            "ties": paired_ties,
        },
        "runs": runs,
    }



def _zero_shot_summary(agent: SparseFlyAgent, task: Task, episodes: int = 32) -> dict:
    from .motor_env import MotorMaze

    solved = 0
    generalized = 0
    best_train = 0.0
    best_hidden = 0.0
    examples: list[str] = []
    for _ in range(episodes):
        env = MotorMaze(task, max_steps=8)
        agent.begin_episode()
        observation = env.observe()
        while not env.done:
            actions = env.valid_actions()
            if not actions:
                break
            action, _ = agent.choose(observation, actions, explore=False)
            next_observation, _, _, _ = env.step(action)
            agent.remember(action)
            observation = next_observation
        train_score = env.last_evaluation.score if env.last_evaluation else 0.0
        hidden_score = env.evaluate_hidden().score if train_score == 1.0 else 0.0
        best_train = max(best_train, train_score)
        best_hidden = max(best_hidden, hidden_score)
        solved += int(train_score == 1.0)
        generalized += int(hidden_score == 1.0)
        if hidden_score == 1.0 and len(examples) < 3:
            examples.append(env.source())
    return {
        "episodes": episodes,
        "solved": solved,
        "generalized": generalized,
        "best_train_score": best_train,
        "best_hidden_score": best_hidden,
        "example_sources": examples,
    }
