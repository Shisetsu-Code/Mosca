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
    def __init__(self, width: int = 8192, hashes_per_token: int = 4):
        self.width = width
        self.hashes_per_token = hashes_per_token

    def _indices(self, token: str) -> Iterable[int]:
        digest = hashlib.blake2b(token.encode(), digest_size=32, person=b"mosca-kc").digest()
        for i in range(self.hashes_per_token):
            start = i * 4
            yield int.from_bytes(digest[start:start + 4], "little") % self.width

    def encode(
        self,
        observation: dict,
        history: tuple[str, ...] = (),
        *,
        include_sensory: bool = True,
    ) -> tuple[int, ...]:
        tokens = [
            f"hole:{observation.get('first_hole')}",
            f"holes:{min(int(observation.get('holes', 0)), 12)}",
            f"nodes:{min(int(observation.get('nodes', 0)) // 2, 16)}",
            f"steps:{min(int(observation.get('steps', 0)) // 3, 16)}",
        ]
        if include_sensory:
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
    """Original fast-policy factorization. Kept stable for baseline behavior."""
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


def memory_action_components(action: str) -> tuple[str, ...]:
    """Semantic factorization used only by task-independent concept memory."""
    keys = [f"exact:{action}"]
    if action.startswith("SET:acc="):
        rhs = action.split("=", 1)[1]
        keys += ("control:write", "dst:acc", "effect:set", f"effect_rhs:{rhs}")
    elif action.startswith("AUG:acc"):
        rest = action[len("AUG:acc"):]
        symbol = rest[0]
        rhs = rest[2:]
        effect = {"+": "add", "-": "sub", "*": "mul"}[symbol]
        keys += ("control:write", "dst:acc", f"effect:{effect}", f"effect_rhs:{rhs}")
    elif action == "FOR:x:xs":
        keys += ("control:loop", "iter:x", "source:xs")
    elif action.startswith("WHEN:"):
        payload = action[len("WHEN:"):]
        cond, update = payload.rsplit(":", 1)
        symbol = "==" if "==" in cond else ">" if ">" in cond else "<"
        left, right = cond.split(symbol, 1)
        effect, effect_rhs = {
            "SETX": ("set", "x"),
            "INC1": ("add", "1"),
            "ADDX": ("add", "x"),
        }[update]
        keys += (
            "control:conditional",
            f"lhs:{left}",
            f"cmp:{symbol}",
            f"cond_rhs:{right}",
            "dst:acc",
            f"effect:{effect}",
            f"effect_rhs:{effect_rhs}",
        )
    elif action == "END":
        keys += ("control:end",)
    elif action.startswith("RETURN:"):
        rhs = action.split(":", 1)[1]
        keys += ("control:return", f"effect_rhs:{rhs}")
    return tuple(keys)


class SparseFlyAgent:
    """Fast TD policy plus exact-context and compositional slow memories."""

    def __init__(self, config: FlyConfig | None = None, seed: int = 0):
        self.config = config or FlyConfig()
        self.encoder = SparseEncoder(self.config.expansion_width, self.config.hashes_per_token)
        self.rng = random.Random(seed)
        self.weights: dict[str, dict[int, float]] = defaultdict(dict)
        self.slow_weights: dict[str, dict[str, dict[int, float]]] = {}
        self.concept_weights: dict[str, dict[str, dict[int, float]]] = {}
        self._active_context = "global"
        self._active_concepts: tuple[str, ...] = ()
        self._last_memory_features: tuple[int, ...] = ()
        self.traces: dict[tuple[str, int], float] = {}
        self.memory_traces: dict[tuple[str, int], float] = {}
        self.history: deque[str] = deque(maxlen=self.config.history)

    def begin_episode(self) -> None:
        self.traces.clear()
        self.memory_traces.clear()
        self.history.clear()

    @staticmethod
    def context_key(observation: dict) -> str:
        sensory = tuple(observation.get("sensory", ()))
        raw = "\x1f".join(sensory).encode()
        return hashlib.blake2b(raw, digest_size=12, person=b"mosca-ctx").hexdigest()

    def features(self, observation: dict) -> tuple[int, ...]:
        return self.encoder.encode(observation, tuple(self.history), include_sensory=True)

    def memory_features(self, observation: dict) -> tuple[int, ...]:
        return self.encoder.encode(observation, tuple(self.history), include_sensory=False)

    def remember(self, action: str) -> None:
        self.history.append(action)

    @staticmethod
    def _table_q(
        table: dict[str, dict[int, float]],
        action: str,
        features: tuple[int, ...],
        component_fn=action_components,
    ) -> float:
        keys = component_fn(action)
        total = 0.0
        for key in keys:
            weights = table.get(key, {})
            total += sum(weights.get(i, 0.0) for i in features)
        return total / math.sqrt(max(1, len(features) * len(keys)))

    def q_fast(self, action: str, features: tuple[int, ...]) -> float:
        return self._table_q(self.weights, action, features, action_components)

    def q(
        self,
        action: str,
        features: tuple[int, ...],
        memory_features: tuple[int, ...],
    ) -> float:
        total = self.q_fast(action, features)
        slow_values: list[float] = []

        context_table = self.slow_weights.get(self._active_context)
        if context_table:
            slow_values.append(
                self._table_q(context_table, action, features, action_components)
            )

        for concept in self._active_concepts:
            table = self.concept_weights.get(concept)
            if table:
                slow_values.append(
                    self._table_q(
                        table, action, memory_features, memory_action_components
                    )
                )

        if slow_values:
            total += self.config.slow_mix * (sum(slow_values) / len(slow_values))
        return total

    def choose(
        self,
        observation: dict,
        valid_actions: tuple[str, ...],
        *,
        explore: bool = True,
    ) -> tuple[str, tuple[int, ...]]:
        if not valid_actions:
            raise ValueError("no valid actions")
        features = self.features(observation)
        memory_features = self.memory_features(observation)
        self._last_memory_features = memory_features
        self._active_context = self.context_key(observation)
        self._active_concepts = tuple(observation.get("memory_concepts", ()))

        if not explore:
            action = max(
                valid_actions,
                key=lambda a: (self.q(a, features, memory_features), a),
            )
            return action, features

        if self.rng.random() < self.config.epsilon:
            return self.rng.choice(valid_actions), features

        scored = [
            (self.q(a, features, memory_features), self.rng.random(), a)
            for a in valid_actions
        ]
        return max(scored)[2], features

    def _consolidate_context(self) -> None:
        rate = self.config.consolidation_rate
        table = self.slow_weights.setdefault(self._active_context, {})
        for (model_key, index), eligibility in self.traces.items():
            if eligibility == 0:
                continue
            fast = self.weights.get(model_key, {}).get(index, 0.0)
            slow_table = table.setdefault(model_key, {})
            slow = slow_table.get(index, 0.0)
            gain = min(1.0, rate * abs(eligibility))
            slow_table[index] = slow + gain * (fast - slow)

    def _consolidate_concepts(self) -> None:
        rate = self.config.consolidation_rate
        for concept in self._active_concepts:
            table = self.concept_weights.setdefault(concept, {})
            for (model_key, index), eligibility in self.memory_traces.items():
                if eligibility <= 0:
                    continue
                weights = table.setdefault(model_key, {})
                old = weights.get(index, 0.0)
                gain = min(1.0, rate * eligibility)
                weights[index] = old + gain * (1.0 - old)

    def learn(
        self,
        features: tuple[int, ...],
        action: str,
        reward: float,
        next_observation: dict,
        next_actions: tuple[str, ...],
        done: bool,
    ) -> float:
        q_now = self.q_fast(action, features)
        if done or not next_actions:
            target = reward
        else:
            next_features = self.encoder.encode(
                next_observation,
                tuple(self.history) + (action,),
                include_sensory=True,
            )
            target = reward + self.config.gamma * max(
                self.q_fast(a, next_features) for a in next_actions
            )
        delta = target - q_now

        decay = self.config.gamma * self.config.trace_decay
        for trace_table in (self.traces, self.memory_traces):
            for key in list(trace_table):
                value = trace_table[key] * decay
                if abs(value) < 1e-5:
                    del trace_table[key]
                else:
                    trace_table[key] = value

        fast_keys = action_components(action)
        memory_keys = memory_action_components(action)
        fast_scale = 1.0 / math.sqrt(max(1, len(features) * len(fast_keys)))
        memory_scale = 1.0 / math.sqrt(
            max(1, len(self._last_memory_features) * len(memory_keys))
        )

        for model_key in fast_keys:
            for i in features:
                key = (model_key, i)
                self.traces[key] = self.traces.get(key, 0.0) + fast_scale

        for model_key in memory_keys:
            for i in self._last_memory_features:
                key = (model_key, i)
                self.memory_traces[key] = self.memory_traces.get(key, 0.0) + memory_scale

        for (model_key, index), eligibility in self.traces.items():
            table = self.weights[model_key]
            table[index] = table.get(index, 0.0) + self.config.alpha * delta * eligibility

        if done and reward >= self.config.consolidation_threshold:
            self._consolidate_context()
            self._consolidate_concepts()

        self.history.append(action)
        return delta

    def fast_parameter_count(self) -> int:
        return sum(len(w) for w in self.weights.values())

    def slow_parameter_count(self) -> int:
        contextual = sum(
            len(weights)
            for context in self.slow_weights.values()
            for weights in context.values()
        )
        conceptual = sum(
            len(weights)
            for concept in self.concept_weights.values()
            for weights in concept.values()
        )
        return contextual + conceptual

    def parameter_count(self) -> int:
        return self.fast_parameter_count() + self.slow_parameter_count()
