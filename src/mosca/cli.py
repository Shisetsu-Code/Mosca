from __future__ import annotations

import argparse
import json

from .ast_env import ASTMaze
from .benchmark import benchmark_json, motor_benchmark, motor_multiseed, transfer_benchmark, transfer_multiseed, transfer_suite
from .grammar import rule_manifest
from .reference import MOTOR_REFERENCE_ACTIONS, REFERENCE_ACTIONS
from .resource import resource_benchmark
from .runtime import runtime_status
from .motor_env import MotorMaze
from .mcts import mcts_benchmark, mcts_multiseed
from .search import bfs_solve
from .tasks import TASKS, TRANSFER_TASKS


def _run_reference(name: str) -> dict:
    env = ASTMaze(TASKS[name], max_steps=64)
    for action in REFERENCE_ACTIONS[name]:
        env.step(action)
    evaluation = env.last_evaluation
    assert evaluation is not None
    return {
        "task": name,
        "actions": len(REFERENCE_ACTIONS[name]),
        "compiled": evaluation.compiled,
        "score": evaluation.score,
        "source": env.source(),
    }


def _run_motor_reference(name: str) -> dict:
    env = MotorMaze(TASKS[name], max_steps=8)
    for action in MOTOR_REFERENCE_ACTIONS[name]:
        env.step(action)
    evaluation = env.last_evaluation
    assert evaluation is not None
    return {
        "task": name,
        "actions": len(env.actions),
        "compiled": evaluation.compiled,
        "score": evaluation.score,
        "source": env.source(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(prog="mosca")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("runtime")
    sub.add_parser("rules")

    solve = sub.add_parser("solve", help="legacy hand-shaped maze baseline")
    solve.add_argument("task", choices=sorted(TASKS))
    solve.add_argument("--depth", type=int, default=10)

    ast_ref = sub.add_parser("ast-reference", help="replay a known solution through the generic AST grammar")
    ast_ref.add_argument("task", choices=["all", *sorted(TASKS)])

    bench = sub.add_parser("benchmark", help="random vs sparse fly learner on the generic AST maze")
    bench.add_argument("--episodes", type=int, default=200)
    bench.add_argument("--seed", type=int, default=0)

    motor_ref = sub.add_parser("motor-reference", help="replay short semantic motor solutions")
    motor_ref.add_argument("task", choices=["all", *sorted(TASKS)])

    motor_bench = sub.add_parser("motor-benchmark", help="random vs fly learner on the hierarchical motor maze")
    motor_bench.add_argument("--episodes", type=int, default=300)
    motor_bench.add_argument("--seed", type=int, default=42)
    motor_bench.add_argument("--require-fly-solved", action="store_true")
    motor_bench.add_argument("--require-fly-generalized", action="store_true")

    multi = sub.add_parser("motor-multiseed", help="repeat motor benchmark across deterministic seeds")
    multi.add_argument("--episodes", type=int, default=200)
    multi.add_argument("--seeds", default="0,1,2")

    transfer = sub.add_parser("transfer-benchmark", help="pretrain on core tasks then adapt to an unseen composition")
    transfer.add_argument("--pretrain-episodes", type=int, default=300)
    transfer.add_argument("--adapt-episodes", type=int, default=200)
    transfer.add_argument("--seed", type=int, default=42)
    transfer.add_argument("--target", choices=sorted(TRANSFER_TASKS), default="sum_positive")

    transfer_multi = sub.add_parser("transfer-multiseed", help="repeat compositional transfer across seeds")
    transfer_multi.add_argument("--pretrain-episodes", type=int, default=600)
    transfer_multi.add_argument("--adapt-episodes", type=int, default=250)
    transfer_multi.add_argument("--seeds", default="0,1,2,3,4")
    transfer_multi.add_argument("--target", choices=sorted(TRANSFER_TASKS), default="sum_positive")

    suite = sub.add_parser("transfer-suite", help="pretrain once and test all held-out compositions")
    suite.add_argument("--pretrain-episodes", type=int, default=600)
    suite.add_argument("--adapt-episodes", type=int, default=250)
    suite.add_argument("--seeds", default="0,1,2,3,4")

    mcts = sub.add_parser("mcts-benchmark", help="UCT baseline over the same MotorMaze")
    mcts.add_argument("--simulations", type=int, default=500)
    mcts.add_argument("--seed", type=int, default=42)

    mcts_multi = sub.add_parser("mcts-multiseed", help="repeat UCT baseline across deterministic seeds")
    mcts_multi.add_argument("--simulations", type=int, default=500)
    mcts_multi.add_argument("--seeds", default="0,1,2,3,4")

    resources = sub.add_parser("resource-benchmark", help="profile cost to first hidden-generalizing solution")
    resources.add_argument("--seed", type=int, default=0)
    resources.add_argument("--pretrain-episodes", type=int, default=600)
    resources.add_argument("--adapt-episodes", type=int, default=250)
    resources.add_argument("--mcts-simulations", type=int, default=500)
    resources.add_argument("--random-episodes", type=int, default=500)

    args = parser.parse_args()

    if args.command == "runtime":
        print(json.dumps(runtime_status(), indent=2))
        return
    if args.command == "rules":
        print(json.dumps(rule_manifest(), indent=2, sort_keys=True))
        return
    if args.command == "ast-reference":
        names = sorted(TASKS) if args.task == "all" else [args.task]
        print(json.dumps([_run_reference(name) for name in names], indent=2))
        return
    if args.command == "benchmark":
        print(benchmark_json(args.episodes, args.seed))
        return
    if args.command == "motor-reference":
        names = sorted(TASKS) if args.task == "all" else [args.task]
        print(json.dumps([_run_motor_reference(name) for name in names], indent=2))
        return
    if args.command == "motor-benchmark":
        result = motor_benchmark(args.episodes, args.seed)
        print(json.dumps(result, indent=2, sort_keys=True))
        if args.require_fly_solved:
            missing = [
                name for name, data in result["tasks"].items()
                if data["fly"]["solved"] == 0
            ]
            if missing:
                raise SystemExit("fly failed to solve: " + ", ".join(missing))
        if args.require_fly_generalized:
            missing = [
                name for name, data in result["tasks"].items()
                if data["fly"]["generalized"] == 0
            ]
            if missing:
                raise SystemExit("fly failed hidden evaluation: " + ", ".join(missing))
        return
    if args.command == "motor-multiseed":
        seeds = tuple(int(x) for x in args.seeds.split(",") if x.strip())
        print(json.dumps(motor_multiseed(args.episodes, seeds), indent=2, sort_keys=True))
        return
    if args.command == "transfer-benchmark":
        print(json.dumps(
            transfer_benchmark(args.pretrain_episodes, args.adapt_episodes, args.seed, args.target),
            indent=2,
            sort_keys=True,
        ))
        return
    if args.command == "transfer-multiseed":
        seeds = tuple(int(x) for x in args.seeds.split(",") if x.strip())
        print(json.dumps(
            transfer_multiseed(args.pretrain_episodes, args.adapt_episodes, seeds, args.target),
            indent=2,
            sort_keys=True,
        ))
        return
    if args.command == "transfer-suite":
        seeds = tuple(int(x) for x in args.seeds.split(",") if x.strip())
        print(json.dumps(
            transfer_suite(args.pretrain_episodes, args.adapt_episodes, seeds),
            indent=2,
            sort_keys=True,
        ))
        return
    if args.command == "mcts-benchmark":
        print(json.dumps(mcts_benchmark(args.simulations, args.seed), indent=2, sort_keys=True))
        return
    if args.command == "mcts-multiseed":
        seeds = tuple(int(x) for x in args.seeds.split(",") if x.strip())
        print(json.dumps(mcts_multiseed(args.simulations, seeds), indent=2, sort_keys=True))
        return
    if args.command == "resource-benchmark":
        print(json.dumps(resource_benchmark(
            seed=args.seed,
            pretrain_episodes=args.pretrain_episodes,
            adapt_episodes=args.adapt_episodes,
            mcts_simulations=args.mcts_simulations,
            random_episodes=args.random_episodes,
        ), indent=2, sort_keys=True))
        return

    result = bfs_solve(TASKS[args.task], max_depth=args.depth)
    if result is None:
        raise SystemExit("No solution found within the requested depth")
    print(f"explored={result.explored} score={result.score:.3f}")
    print("actions=" + ",".join(a.value for a in result.actions))
    print(result.source)


if __name__ == "__main__":
    main()
