from __future__ import annotations

import argparse
import json

from .runtime import runtime_status
from .search import bfs_solve
from .tasks import TASKS


def main() -> None:
    parser = argparse.ArgumentParser(prog="mosca")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("runtime")
    solve = sub.add_parser("solve")
    solve.add_argument("task", choices=sorted(TASKS))
    solve.add_argument("--depth", type=int, default=10)
    args = parser.parse_args()

    if args.command == "runtime":
        print(json.dumps(runtime_status(), indent=2))
        return

    result = bfs_solve(TASKS[args.task], max_depth=args.depth)
    if result is None:
        raise SystemExit("No solution found within the requested depth")
    print(f"explored={result.explored} score={result.score:.3f}")
    print("actions=" + ",".join(a.value for a in result.actions))
    print(result.source)


if __name__ == "__main__":
    main()
