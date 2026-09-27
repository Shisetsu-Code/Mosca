from __future__ import annotations

import hashlib
import math
import random
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class FlyConfig:
    expansion_width: int = 8192
    hashes_per_token: int = 4
    history: int = 4
    alpha: float = 0.08
    gamma: float = 0.97
    trace_decay: float = 0.85
    epsilon: float = 0.18
    consolidation_rate: float = 0.0
    slow_mix: float = 0.0
    consolidation_threshold: float = 7.0


class SparseEncoder:
    """Deterministic sparse expansion code, analogous in function to KC expansion."""

    def __init__(self, width: int = 8192, hashes_per_token: int = 4):
        self.width = width
        self.hashes_per_token = hashes_per_token

    def _indices(self, token: str) -> Iterable[int]:
        digest = hashlib.blake2b(token.encode(), digest_size=32, person=b"mosca-kc").digest()
        for i in range(self.hashes_per_token):
            start = i * 4
            yield int.from_bytes(digest[start:start + 4], "little") % self.width

    def encode(self, observation: dict, history: tuple[str, ...] = ()) -> tuple[int, ...]:
        tokens = [
            f"hole:{observation.get('first_hole')}",
            f"holes:{min(int(observation.get('holes', 0)), 12)}",
            f"nodes:{min(int(observation.get('nodes', 0)) // 2, 16)}",
            f"steps:{min(int(observation.get('steps', 0)) // 3, 16)}",
        ]
        for token in observation.get("sensory", ()):
            tokens.append(f"sense:{token}")
        for tag, count in sorted(observation.get("tags", {}).items()):
            tokens.append(f"tag:{tag}:{min(int(count), 4)}")
        for pos, action in enumerate(reversed(history)):
            tokens.append(f"hist:{pos}:{action}")
        active: set[int] = set()
        for token in tokens:
            active.update(self._indices(token))
        return tuple(sorted(active))


def action_components(action: str) -> tuple[str, ...]:
    """Factor a motor action so learned parts can recombine on unseen tasks."""

    keys = [f"exact:{action}"]
    if action.startswith("SET:acc="):
        rhs = action.split("=", 1)[1]
        rhs_kind = "const" if rhs.lstrip("-").isdigit() else "input" if rhs.startswith("xs") else "var"
        keys += ("op:SET", "dst:acc", f"rhs_kind:{rhs_kind}", f"rhs:{rhs}")
    elif action.startswith("AUG:acc"):
        rest = action[len("AUG:acc"):]
        symbol = rest[0]
        rhs = rest[2:]
        keys += ("op:AUG", "dst:acc", f"arith:{symbol}", f"rhs:{rhs}")
    elif action == "FOR:x:xs":
        keys += ("op:FOR", "iter:x", "source:xs")
    elif action.startswith("WHEN:"):
        payload = action[len("WHEN:"):]
        cond, update = payload.rsplit(":", 1)
        symbol = "==" if "==" in cond else ">" if ">" in cond else "<"
        left, right = cond.split(symbol, 1)
        keys += (
            "op:WHEN",
            f"lhs:{left}",
            f"cmp:{symbol}",
            f"cond_rhs:{right}",
            f"update:{update}",
        )
    elif action == "END":
        keys += ("op:END",)
    elif action.startswith("RETURN:"):
        keys += ("op:RETURN", f"rhs:{action.split(':', 1)[1]}")
    return tuple(keys)


class SparseFlyAgent:
    """Sparse recurrent TD(lambda) policy with optional slow synaptic consolidation."""

    def __init__(self, config: FlyConfig | None = None, seed: int = 0):
        self.config = config or FlyConfig()
        self.encoder = SparseEncoder(self.config.expansion_width, self.config.hashes_per_token)
        self.rng = random.Random(seed)
        self.weights: dict[str, dict[int, float]] = defaultdict(dict)
        self.slow_weights: dict[str, dict[str, dict[int, float]]] = {}\n        self._active_context = "global"
        self.traces: dict[tuple[str, int], float] = {}
        self.history: deque[str] = deque(maxlen=self.config.history)

    def begin_episode(self) -> None:
        self.traces.clear()
        self.history.clear()

    @staticmethod
    def context_key(observation: dict) -> str:
        sensory = tuple(observation.get("sensory", ()))
        raw = "\x1f".join(sensory).encode()
        return hashlib.blake2b(raw, digest_size=12, person=b"mosca-ctx").hexdigest()

    def features(self, observation: dict) -> tuple[int, ...]:
        return self.encoder.encode(observation, tuple(self.history))

    def q(self, action: str, features: tuple[int, ...], context: str | None = None) -> float:
        keys = action_components(action)
        total = 0.0
        context = self._active_context if context is None else context
        context_slow = self.slow_weights.get(context, {})
        for key in keys:
            fast = self.weights.get(key, {})
            slow = context_slow.get(key, {})
            for i in features:
                total += fast.get(i, 0.0)
                total += self.config.slow_mix * slow.get(i, 0.0)
        return total / math.sqrt(max(1, len(features) * len(keys)))

    def choose(self, observation: dict, valid_actions: tuple[str, ...], *, explore: bool = True) -> tuple[str, tuple[int, ...]]:
        if not valid_actions:
            raise ValueError("no valid actions")
        features = self.features(observation)
        self._active_context = self.context_key(observation)
        if explore and self.rng.random() < self.config.epsilon:
            return self.rng.choice(valid_actions), features
        scored = [(self.q(a, features, self._active_context), self.rng.random(), a) for a in valid_actions]
        return max(scored)[2], features

    def _consolidate(self, context: str) -> None:
        rate = self.config.consolidation_rate
        if rate <= 0:
            return
        context_table = self.slow_weights.setdefault(context, {})
        for (model_key, index), eligibility in self.traces.items():
            if eligibility == 0:
                continue
            fast = self.weights.get(model_key, {}).get(index, 0.0)
            slow_table = context_table.setdefault(model_key, {})
            slow = slow_table.get(index, 0.0)
            gain = min(1.0, rate * abs(eligibility))
            slow_table[index] = slow + gain * (fast - slow)

    def learn(
        self,
        features: tuple[int, ...],
        action: str,
        reward: float,
        next_observation: dict,
        next_actions: tuple[str, ...],
        done: bool,
    ) -> float:
        context = self._active_context
        q_now = self.q(action, features, context)
        if done or not next_actions:
            target = reward
        else:
            next_context = self.context_key(next_observation)
            next_features = self.encoder.encode(next_observation, tuple(self.history) + (action,))
            target = reward + self.config.gamma * max(self.q(a, next_features, next_context) for a in next_actions)
        delta = target - q_now

        decay = self.config.gamma * self.config.trace_decay
        for key in list(self.traces):
            value = self.traces[key] * decay
            if abs(value) < 1e-5:
                del self.traces[key]
            else:
                self.traces[key] = value

        keys = action_components(action)
        scale = 1.0 / math.sqrt(max(1, len(features) * len(keys)))
        for model_key in keys:
            for i in features:
                trace_key = (model_key, i)
                self.traces[trace_key] = self.traces.get(trace_key, 0.0) + scale

        for (model_key, index), eligibility in self.traces.items():
            table = self.weights[model_key]
            table[index] = table.get(index, 0.0) + self.config.alpha * delta * eligibility

        if done and reward >= self.config.consolidation_threshold:
            self._consolidate(context)

        self.history.append(action)
        return delta

    def fast_parameter_count(self) -> int:
        return sum(len(w) for w in self.weights.values())

    def slow_parameter_count(self) -> int:
        return sum(
            len(weights)
            for context in self.slow_weights.values()
            for weights in context.values()
        )

    def parameter_count(self) -> int:
        return self.fast_parameter_count() + self.slow_parameter_count()
