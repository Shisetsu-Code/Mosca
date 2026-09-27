from __future__ import annotations

from collections import Counter

from .tasks import Task


def _bucket_int(value: int) -> str:
    if value == 0:
        return "0"
    sign = "p" if value > 0 else "n"
    mag = abs(value)
    if mag == 1:
        size = "1"
    elif mag <= 3:
        size = "2-3"
    elif mag <= 8:
        size = "4-8"
    else:
        size = "9+"
    return f"{sign}:{size}"


def _case_features(xs: list[int], out: int) -> set[str]:
    features = {
        f"len:{min(len(xs), 5)}",
        f"out:{_bucket_int(out)}",
        f"pos:{min(sum(x > 0 for x in xs), 4)}",
        f"neg:{min(sum(x < 0 for x in xs), 4)}",
        f"zero:{min(sum(x == 0 for x in xs), 3)}",
    }
    total = sum(xs)
    pos_sum = sum(x for x in xs if x > 0)
    neg_sum = sum(x for x in xs if x < 0)
    pos_count = sum(x > 0 for x in xs)
    neg_count = sum(x < 0 for x in xs)

    relations = {
        "eq:sum": out == total,
        "eq:pos_sum": out == pos_sum,
        "eq:neg_sum": out == neg_sum,
        "eq:len": out == len(xs),
        "eq:pos_count": out == pos_count,
        "eq:neg_count": out == neg_count,
        "eq:max": bool(xs) and out == max(xs),
        "eq:min": bool(xs) and out == min(xs),
        "member": out in xs if xs else False,
        "out>=0": out >= 0,
    }
    features.update(k for k, v in relations.items() if v)
    if xs:
        features.add(f"rank:{'hi' if out >= max(xs) else 'lo' if out <= min(xs) else 'mid'}")
    return features


def task_sensory_tokens(task: Task) -> tuple[str, ...]:
    """Compact, deterministic sensory signature derived only from visible I/O cases.

    This replaces the task-name token. The learner sees low-order properties of
    examples, not natural-language descriptions or target source code.
    """

    case_sets: list[set[str]] = []
    for case in task.cases:
        xs = list(case.args[0])
        case_sets.append(_case_features(xs, int(case.expected)))

    counts = Counter(token for features in case_sets for token in features)
    n = max(1, len(case_sets))
    tokens: list[str] = []
    for token, count in sorted(counts.items()):
        if count == n:
            freq = "all"
        elif count * 2 >= n:
            freq = "many"
        else:
            freq = "some"
        tokens.append(f"{freq}:{token}")
    return tuple(tokens)
