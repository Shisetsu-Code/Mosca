from __future__ import annotations

import hashlib
import math
import random
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Iterable

import numpy as np


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
    role_factor_mix: float = 0.0
    agg_factor_mix: float = 0.0
    filter_factor_mix: float = 0.0
    adaptive_factor_gates: bool = False
    gate_alpha: float = 0.02
    gate_trace_decay: float = 0.90
    consolidation_threshold: float = 7.0


class SparseEncoder:
    def __init__(self, width: int = 8192, hashes_per_token: int = 4):
        self.width = width
        self.hashes_per_token = hashes_per_token
        self._index_cache: dict[str, tuple[int, ...]] = {}

    def _indices(self, token: str) -> tuple[int, ...]:
        cached = self._index_cache.get(token)
        if cached is not None:
            return cached
        digest = hashlib.blake2b(token.encode(), digest_size=32, person=b"mosca-kc").digest()
        indices = tuple(
            int.from_bytes(digest[i * 4:i * 4 + 4], "little") % self.width
            for i in range(self.hashes_per_token)
        )
        self._index_cache[token] = indices
        return indices

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


def factor_memory_components(action: str) -> tuple[str, ...]:
    """Factor-only motor semantics; never used by the established broad memory."""
    keys: list[str] = []
    if action.startswith("SET:acc="):
        rhs = action.split("=", 1)[1]
        keys += ("control:init", "dst:acc", "effect:set", f"effect_rhs:{rhs}")
    elif action.startswith("AUG:acc"):
        rest = action[len("AUG:acc"):]
        symbol = rest[0]
        rhs = rest[2:]
        effect = {"+": "add", "-": "sub", "*": "mul"}[symbol]
        keys += (
            "control:unconditional", "dst:acc",
            f"effect:{effect}", f"effect_rhs:{rhs}",
        )
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
            "control:conditional", f"lhs:{left}", f"cmp:{symbol}",
            f"cond_rhs:{right}", "dst:acc",
            f"effect:{effect}", f"effect_rhs:{effect_rhs}",
        )
    elif action == "END":
        keys += ("control:end",)
    elif action.startswith("RETURN:"):
        rhs = action.split(":", 1)[1]
        keys += ("control:return", f"effect_rhs:{rhs}")
    return tuple(keys)


_CONCEPT_FACTOR_COMPONENTS: dict[str, frozenset[str]] = {
    "agg:sum": frozenset({"effect:add", "effect_rhs:x"}),
    "agg:count": frozenset({"effect:add", "effect_rhs:1"}),
    "agg:max": frozenset({
        "control:conditional", "lhs:x", "cmp:>", "cond_rhs:acc",
        "effect:set", "effect_rhs:x",
    }),
    "agg:min": frozenset({
        "control:conditional", "lhs:x", "cmp:<", "cond_rhs:acc",
        "effect:set", "effect_rhs:x",
    }),
    "filter:positive": frozenset({
        "control:conditional", "lhs:x", "cmp:>", "cond_rhs:0",
    }),
    "filter:negative": frozenset({
        "control:conditional", "lhs:x", "cmp:<", "cond_rhs:0",
    }),
    "filter:all": frozenset({"control:unconditional"}),
    "role:identity_zero": frozenset({
        "control:init", "dst:acc", "effect:set", "effect_rhs:0",
    }),
    "role:list_reduce": frozenset({
        "control:loop", "iter:x", "source:xs",
        "control:end", "control:return", "effect_rhs:acc",
    }),
}


def concept_accepts_factor(concept: str, component: str) -> bool:
    return component in _CONCEPT_FACTOR_COMPONENTS.get(concept, frozenset())


class SparseFlyAgent:
    """Fast TD policy plus exact-context and compositional slow memories."""

    def __init__(self, config: FlyConfig | None = None, seed: int = 0):
        self.config = config or FlyConfig()
        self.encoder = SparseEncoder(self.config.expansion_width, self.config.hashes_per_token)
        self.rng = random.Random(seed)
        self.weights: dict[str, np.ndarray] = {}
        self.slow_weights: dict[str, dict[str, dict[int, float]]] = {}
        self.concept_weights: dict[str, dict[str, dict[int, float]]] = {}
        self.factor_concept_weights: dict[str, dict[str, dict[int, float]]] = {}
        self._active_context = "global"
        self._active_concepts: tuple[str, ...] = ()
        self._last_memory_features: tuple[int, ...] = ()
        self.traces: dict[str, np.ndarray] = {}
        self.memory_traces: dict[str, np.ndarray] = {}
        self.factor_memory_traces: dict[str, np.ndarray] = {}
        self.factor_gates: dict[str, float] = {
            "role": float(self.config.role_factor_mix),
            "agg": float(self.config.agg_factor_mix),
            "filter": float(self.config.filter_factor_mix),
        }
        self.gate_traces: dict[str, float] = {
            "role": 0.0,
            "agg": 0.0,
            "filter": 0.0,
        }
        self._last_factor_values: dict[str, float] = {
            "role": 0.0,
            "agg": 0.0,
            "filter": 0.0,
        }
        self._adaptive_was_enabled = bool(self.config.adaptive_factor_gates)
        self.history: deque[str] = deque(maxlen=self.config.history)

    def _sync_factor_gates(self) -> None:
        enabled = bool(self.config.adaptive_factor_gates)
        if enabled and not self._adaptive_was_enabled:
            self.factor_gates = {
                "role": float(self.config.role_factor_mix),
                "agg": float(self.config.agg_factor_mix),
                "filter": float(self.config.filter_factor_mix),
            }
        self._adaptive_was_enabled = enabled

    def begin_episode(self) -> None:
        self._sync_factor_gates()
        self.traces.clear()
        self.memory_traces.clear()
        self.factor_memory_traces.clear()
        self.gate_traces = {"role": 0.0, "agg": 0.0, "filter": 0.0}
        self._last_factor_values = {"role": 0.0, "agg": 0.0, "filter": 0.0}
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

    def _fast_table(self, key: str) -> np.ndarray:
        table = self.weights.get(key)
        if table is None:
            table = np.zeros(self.config.expansion_width, dtype=np.float64)
            self.weights[key] = table
        return table

    def q_fast(self, action: str, features: tuple[int, ...]) -> float:
        keys = action_components(action)
        indices = np.asarray(features, dtype=np.intp)
        total = 0.0
        for key in keys:
            weights = self.weights.get(key)
            if weights is not None:
                total += float(weights[indices].sum())
        return total / math.sqrt(max(1, len(features) * len(keys)))

    def q_fast_many(
        self,
        actions: tuple[str, ...],
        features: tuple[int, ...],
    ) -> dict[str, float]:
        """Evaluate shared action components once for one sparse state."""
        if not actions:
            return {}
        keys_by_action = {action: action_components(action) for action in actions}
        unique_keys = {
            key
            for keys in keys_by_action.values()
            for key in keys
        }
        indices = np.asarray(features, dtype=np.intp)
        component_scores: dict[str, float] = {}
        for key in unique_keys:
            weights = self.weights.get(key)
            component_scores[key] = (
                float(weights[indices].sum()) if weights is not None else 0.0
            )

        feature_count = len(features)
        return {
            action: (
                sum(component_scores[key] for key in keys)
                / math.sqrt(max(1, feature_count * len(keys)))
            )
            for action, keys in keys_by_action.items()
        }

    @staticmethod
    def _factor_table_q(
        concept: str,
        table: dict[str, dict[int, float]],
        action: str,
        features: tuple[int, ...],
    ) -> float:
        keys = tuple(
            key for key in factor_memory_components(action)
            if concept_accepts_factor(concept, key)
        )
        if not keys:
            return 0.0
        total = 0.0
        for key in keys:
            weights = table.get(key, {})
            total += sum(weights.get(i, 0.0) for i in features)
        return total / math.sqrt(max(1, len(features) * len(keys)))

    @staticmethod
    def _factor_family(concept: str) -> str | None:
        if concept.startswith("role:"):
            return "role"
        if concept.startswith("agg:"):
            return "agg"
        if concept.startswith("filter:"):
            return "filter"
        return None

    def factor_gate_snapshot(self) -> dict[str, float]:
        self._sync_factor_gates()
        if self.config.adaptive_factor_gates:
            return dict(self.factor_gates)
        return {
            "role": float(self.config.role_factor_mix),
            "agg": float(self.config.agg_factor_mix),
            "filter": float(self.config.filter_factor_mix),
        }

    def _factor_family_values(
        self,
        action: str,
        memory_features: tuple[int, ...],
    ) -> dict[str, float]:
        values: dict[str, list[float]] = {
            "role": [],
            "agg": [],
            "filter": [],
        }
        for concept in self._active_concepts:
            family = self._factor_family(concept)
            if family is None:
                continue
            table = self.factor_concept_weights.get(concept)
            if not table:
                continue
            value = self._factor_table_q(
                concept, table, action, memory_features
            )
            if value != 0.0:
                values[family].append(value)
        return {
            family: (sum(items) / len(items) if items else 0.0)
            for family, items in values.items()
        }

    def _factor_mix(self, family: str) -> float:
        self._sync_factor_gates()
        if self.config.adaptive_factor_gates:
            value = self.factor_gates[family]
        elif family == "role":
            value = self.config.role_factor_mix
        elif family == "agg":
            value = self.config.agg_factor_mix
        else:
            value = self.config.filter_factor_mix
        return min(1.0, max(0.0, float(value)))

    def _q_with_fast(
        self,
        action: str,
        fast_value: float,
        features: tuple[int, ...],
        memory_features: tuple[int, ...],
    ) -> float:
        total = fast_value
        slow_values: list[float] = []

        context_table = self.slow_weights.get(self._active_context)
        if context_table:
            slow_values.append(
                self._table_q(context_table, action, features, action_components)
            )

        for concept in self._active_concepts:
            if concept.startswith("role:"):
                continue
            table = self.concept_weights.get(concept)
            if table:
                slow_values.append(
                    self._table_q(
                        table, action, memory_features, memory_action_components
                    )
                )

        if slow_values:
            total += self.config.slow_mix * (sum(slow_values) / len(slow_values))

        family_values = self._factor_family_values(
            action, memory_features
        )
        weighted = [
            self._factor_mix(family) * value
            for family, value in family_values.items()
            if value != 0.0 and self._factor_mix(family) > 0.0
        ]
        if weighted:
            total += self.config.slow_mix * (
                sum(weighted) / len(weighted)
            )
        return total

    def q(
        self,
        action: str,
        features: tuple[int, ...],
        memory_features: tuple[int, ...],
    ) -> float:
        return self._q_with_fast(
            action,
            self.q_fast(action, features),
            features,
            memory_features,
        )

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
            fast_scores = self.q_fast_many(valid_actions, features)
            action = max(
                valid_actions,
                key=lambda a: (
                    self._q_with_fast(
                        a, fast_scores[a], features, memory_features
                    ),
                    a,
                ),
            )
            self._last_factor_values = self._factor_family_values(
                action, memory_features
            )
            return action, features

        if self.rng.random() < self.config.epsilon:
            action = self.rng.choice(valid_actions)
            self._last_factor_values = self._factor_family_values(
                action, memory_features
            )
            return action, features

        fast_scores = self.q_fast_many(valid_actions, features)
        scored = [
            (
                self._q_with_fast(
                    a, fast_scores[a], features, memory_features
                ),
                self.rng.random(),
                a,
            )
            for a in valid_actions
        ]
        action = max(scored)[2]
        self._last_factor_values = self._factor_family_values(
            action, memory_features
        )
        return action, features

    def _consolidate_context(self) -> None:
        rate = self.config.consolidation_rate
        table = self.slow_weights.setdefault(self._active_context, {})
        for model_key, eligibility in self.traces.items():
            fast_table = self.weights.get(model_key)
            if fast_table is None:
                continue
            slow_table = table.setdefault(model_key, {})
            for raw_index in np.flatnonzero(eligibility):
                index = int(raw_index)
                trace_value = float(eligibility[index])
                fast = float(fast_table[index])
                slow = slow_table.get(index, 0.0)
                gain = min(1.0, rate * abs(trace_value))
                slow_table[index] = slow + gain * (fast - slow)

    def _consolidate_concepts(self) -> None:
        rate = self.config.consolidation_rate

        # Established v0.9 broad memory: unchanged, and role concepts are
        # deliberately excluded so role_mix=0 is behaviorally identical.
        for concept in self._active_concepts:
            if concept.startswith("role:"):
                continue
            table = self.concept_weights.setdefault(concept, {})
            for model_key, eligibility in self.memory_traces.items():
                weights = table.setdefault(model_key, {})
                for raw_index in np.flatnonzero(eligibility):
                    index = int(raw_index)
                    trace_value = float(eligibility[index])
                    if trace_value <= 0:
                        continue
                    old = weights.get(index, 0.0)
                    gain = min(1.0, rate * trace_value)
                    weights[index] = old + gain * (1.0 - old)

        # Independent factor channel for all semantic concepts. It learns
        # silently while its per-family gate is zero.
        for concept in self._active_concepts:
            table = self.factor_concept_weights.setdefault(concept, {})
            for model_key, eligibility in self.factor_memory_traces.items():
                if not concept_accepts_factor(concept, model_key):
                    continue
                weights = table.setdefault(model_key, {})
                for raw_index in np.flatnonzero(eligibility):
                    index = int(raw_index)
                    trace_value = float(eligibility[index])
                    if trace_value <= 0:
                        continue
                    old = weights.get(index, 0.0)
                    gain = min(1.0, rate * trace_value)
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
            next_scores = self.q_fast_many(next_actions, next_features)
            target = reward + self.config.gamma * max(next_scores.values())
        delta = target - q_now

        if self.config.adaptive_factor_gates:
            gate_decay = (
                self.config.gamma * self.config.gate_trace_decay
            )
            modulator = math.tanh(delta)
            for family, signal in self._last_factor_values.items():
                trace = (
                    gate_decay * self.gate_traces[family]
                    + math.tanh(signal)
                )
                self.gate_traces[family] = trace
                updated = (
                    self.factor_gates[family]
                    + self.config.gate_alpha * modulator * trace
                )
                self.factor_gates[family] = min(
                    1.0, max(0.0, updated)
                )

        decay = self.config.gamma * self.config.trace_decay
        for eligibility in self.traces.values():
            eligibility *= decay
        for eligibility in self.memory_traces.values():
            eligibility *= decay
        for eligibility in self.factor_memory_traces.values():
            eligibility *= decay

        fast_keys = action_components(action)
        memory_keys = memory_action_components(action)
        factor_keys = factor_memory_components(action)
        fast_scale = 1.0 / math.sqrt(max(1, len(features) * len(fast_keys)))
        memory_scale = 1.0 / math.sqrt(
            max(1, len(self._last_memory_features) * len(memory_keys))
        )
        factor_scale = 1.0 / math.sqrt(
            max(1, len(self._last_memory_features) * max(1, len(factor_keys)))
        )
        fast_indices = np.asarray(features, dtype=np.intp)
        memory_indices = np.asarray(self._last_memory_features, dtype=np.intp)

        for model_key in fast_keys:
            eligibility = self.traces.get(model_key)
            if eligibility is None:
                eligibility = np.zeros(self.config.expansion_width, dtype=np.float64)
                self.traces[model_key] = eligibility
            eligibility[fast_indices] += fast_scale

        for model_key in memory_keys:
            eligibility = self.memory_traces.get(model_key)
            if eligibility is None:
                eligibility = np.zeros(self.config.expansion_width, dtype=np.float64)
                self.memory_traces[model_key] = eligibility
            eligibility[memory_indices] += memory_scale

        for model_key in factor_keys:
            eligibility = self.factor_memory_traces.get(model_key)
            if eligibility is None:
                eligibility = np.zeros(self.config.expansion_width, dtype=np.float64)
                self.factor_memory_traces[model_key] = eligibility
            eligibility[memory_indices] += factor_scale

        update_scale = self.config.alpha * delta
        for model_key, eligibility in self.traces.items():
            table = self._fast_table(model_key)
            table += update_scale * eligibility

        if done and reward >= self.config.consolidation_threshold:
            self._consolidate_context()
            self._consolidate_concepts()

        self.history.append(action)
        return delta

    def fast_parameter_count(self) -> int:
        return sum(int(np.count_nonzero(w)) for w in self.weights.values())

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
        factor_conceptual = sum(
            len(weights)
            for concept in self.factor_concept_weights.values()
            for weights in concept.values()
        )
        return contextual + conceptual + factor_conceptual

    def parameter_count(self) -> int:
        return self.fast_parameter_count() + self.slow_parameter_count()
