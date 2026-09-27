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
    relations = {
        "eq:sum": out == sum(xs),
        "eq:pos_sum": out == sum(x for x in xs if x > 0),
        "eq:neg_sum": out == sum(x for x in xs if x < 0),
        "eq:len": out == len(xs),
        "eq:pos_count": out == sum(x > 0 for x in xs),
        "eq:neg_count": out == sum(x < 0 for x in xs),
        "eq:max": bool(xs) and out == max(xs),
        "eq:min": bool(xs) and out == min(xs),
        "member": out in xs if xs else False,
        "out>=0": out >= 0,
    }
    features.update(k for k, v in relations.items() if v)
    if xs:
        features.add(f"rank:{'hi' if out >= max(xs) else 'lo' if out <= min(xs) else 'mid'}")
    return features


def _feature_counts(task: Task) -> tuple[Counter[str], int]:
    case_sets = [
        _case_features(list(case.args[0]), int(case.expected))
        for case in task.cases
    ]
    return Counter(token for features in case_sets for token in features), max(1, len(case_sets))


def task_sensory_tokens(task: Task) -> tuple[str, ...]:
    """Fast sensory signature derived only from visible I/O cases."""

    counts, n = _feature_counts(task)
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


def task_memory_concepts(task: Task) -> tuple[str, ...]:
    """Stable semantic axes used only to route slow memory.

    These are inferred from visible I/O relations and are intentionally kept out
    of the fast sparse encoder.
    """

    counts, n = _feature_counts(task)
    stable = {token for token, count in counts.items() if count == n}
    mapping: dict[str, tuple[str, ...]] = {
        "eq:sum": ("agg:sum", "filter:all"),
        "eq:pos_sum": ("agg:sum", "filter:positive"),
        "eq:neg_sum": ("agg:sum", "filter:negative"),
        "eq:len": ("agg:count", "filter:all"),
        "eq:pos_count": ("agg:count", "filter:positive"),
        "eq:neg_count": ("agg:count", "filter:negative"),
        "eq:max": ("agg:max", "filter:all"),
        "eq:min": ("agg:min", "filter:all"),
    }
    concepts: set[str] = set()
    for relation in stable:
        concepts.update(mapping.get(relation, ()))
    return tuple(sorted(concepts))
