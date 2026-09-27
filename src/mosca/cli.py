from __future__ import annotations

import argparse
import json

from .ast_env import ASTMaze
from .benchmark import benchmark_json, motor_benchmark
from .grammar import rule_manifest
from .reference import MOTOR_REFERENCE_ACTIONS, REFERENCE_ACTIONS
from .runtime import runtime_status
from .motor_env import MotorMaze
from .search import bfs_solve
from .tasks import TASKS


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

    result = bfs_solve(TASKS[args.task], max_depth=args.depth)
    if result is None:
        raise SystemExit("No solution found within the requested depth")
    print(f"explored={result.explored} score={result.score:.3f}")
    print("actions=" + ",".join(a.value for a in result.actions))
    print(result.source)


if __name__ == "__main__":
    main()
