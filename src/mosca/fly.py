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
            f"task:{observation.get('task')}",
            f"hole:{observation.get('first_hole')}",
            f"holes:{min(int(observation.get('holes', 0)), 12)}",
            f"nodes:{min(int(observation.get('nodes', 0)) // 2, 16)}",
            f"steps:{min(int(observation.get('steps', 0)) // 3, 16)}",
        ]
        for tag, count in sorted(observation.get("tags", {}).items()):
            tokens.append(f"tag:{tag}:{min(int(count), 4)}")
        for pos, action in enumerate(reversed(history)):
            tokens.append(f"hist:{pos}:{action}")
        active: set[int] = set()
        for token in tokens:
            active.update(self._indices(token))
        return tuple(sorted(active))


class SparseFlyAgent:
    """Sparse recurrent TD(lambda) policy with reward-modulated eligibility traces."""

    def __init__(self, config: FlyConfig | None = None, seed: int = 0):
        self.config = config or FlyConfig()
        self.encoder = SparseEncoder(self.config.expansion_width, self.config.hashes_per_token)
        self.rng = random.Random(seed)
        self.weights: dict[str, dict[int, float]] = defaultdict(dict)
        self.traces: dict[tuple[str, int], float] = {}
        self.history: deque[str] = deque(maxlen=self.config.history)

    def begin_episode(self) -> None:
        self.traces.clear()
        self.history.clear()

    def features(self, observation: dict) -> tuple[int, ...]:
        return self.encoder.encode(observation, tuple(self.history))

    def q(self, action: str, features: tuple[int, ...]) -> float:
        weights = self.weights.get(action, {})
        return sum(weights.get(i, 0.0) for i in features) / math.sqrt(max(1, len(features)))

    def choose(self, observation: dict, valid_actions: tuple[str, ...], *, explore: bool = True) -> tuple[str, tuple[int, ...]]:
        if not valid_actions:
            raise ValueError("no valid actions")
        features = self.features(observation)
        if explore and self.rng.random() < self.config.epsilon:
            return self.rng.choice(valid_actions), features
        scored = [(self.q(a, features), self.rng.random(), a) for a in valid_actions]
        return max(scored)[2], features

    def learn(
        self,
        features: tuple[int, ...],
        action: str,
        reward: float,
        next_observation: dict,
        next_actions: tuple[str, ...],
        done: bool,
    ) -> float:
        q_now = self.q(action, features)
        if done or not next_actions:
            target = reward
        else:
            next_features = self.encoder.encode(next_observation, tuple(self.history) + (action,))
            target = reward + self.config.gamma * max(self.q(a, next_features) for a in next_actions)
        delta = target - q_now

        decay = self.config.gamma * self.config.trace_decay
        for key in list(self.traces):
            value = self.traces[key] * decay
            if abs(value) < 1e-5:
                del self.traces[key]
            else:
                self.traces[key] = value

        scale = 1.0 / math.sqrt(max(1, len(features)))
        for i in features:
            self.traces[(action, i)] = self.traces.get((action, i), 0.0) + scale

        for (trace_action, index), eligibility in self.traces.items():
            table = self.weights[trace_action]
            table[index] = table.get(index, 0.0) + self.config.alpha * delta * eligibility

        self.history.append(action)
        return delta

    def parameter_count(self) -> int:
        return sum(len(w) for w in self.weights.values())
