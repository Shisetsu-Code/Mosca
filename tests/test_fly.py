from mosca.ast_env import ASTMaze
from mosca.fly import (
    FlyConfig,
    SparseEncoder,
    SparseFlyAgent,
    action_components,
    concept_accepts_factor,
    factor_memory_components,
    memory_action_components,
)
from mosca.motor_env import MotorMaze
from mosca.tasks import TASKS, TRANSFER_TASKS


def test_sparse_encoder_is_deterministic_and_sparse():
    obs = MotorMaze(TASKS["sum_list"]).observe()
    enc = SparseEncoder(width=1024, hashes_per_token=3)
    a = enc.encode(obs)
    b = enc.encode(obs)
    assert a == b
    assert 0 < len(a) < 300


def test_fast_action_factorization_stays_baseline_compatible():
    plain = set(action_components("AUG:acc+=x"))
    assert "op:AUG" in plain
    assert "effect:add" not in plain


def test_memory_effects_bridge_plain_and_conditional_updates():
    plain = set(memory_action_components("AUG:acc+=x"))
    conditional = set(memory_action_components("WHEN:x>0:ADDX"))
    assert {"effect:add", "effect_rhs:x", "dst:acc"} <= plain & conditional


def test_memory_features_ignore_task_sensory_signature():
    agent = SparseFlyAgent(seed=1)
    a = MotorMaze(TASKS["sum_list"]).observe()
    b = MotorMaze(TRANSFER_TASKS["sum_positive"]).observe()
    # At the same motor state the shared memory representation is identical.
    assert agent.memory_features(a) == agent.memory_features(b)
    assert agent.features(a) != agent.features(b)


def test_success_consolidates_shared_concept_memory():
    agent = SparseFlyAgent(
        FlyConfig(consolidation_rate=1.0, slow_mix=0.5, consolidation_threshold=-1.0),
        seed=3,
    )
    env = MotorMaze(TASKS["sum_list"])
    obs = env.observe()
    assert "agg:sum" in obs["memory_concepts"]
    agent.begin_episode()
    action, features = agent.choose(obs, env.valid_actions())
    next_obs, reward, _, _ = env.step(action)
    agent.learn(features, action, max(reward, 0.0), next_obs, env.valid_actions(), True)
    assert "agg:sum" in agent.concept_weights
    assert agent.slow_parameter_count() > 0


def test_fly_agent_still_updates_fast_weights():
    env = ASTMaze(TASKS["sum_list"])
    agent = SparseFlyAgent(seed=1)
    agent.begin_episode()
    obs = env.observe()
    action, features = agent.choose(obs, env.valid_actions())
    next_obs, reward, done, _ = env.step(action)
    agent.learn(features, action, reward, next_obs, env.valid_actions(), done)
    assert agent.fast_parameter_count() > 0


def test_fast_weights_use_dense_numpy_vectors():
    import numpy as np

    agent = SparseFlyAgent(seed=4)
    env = ASTMaze(TASKS["sum_list"])
    obs = env.observe()
    action, features = agent.choose(obs, env.valid_actions())
    next_obs, reward, done, _ = env.step(action)
    agent.learn(features, action, reward, next_obs, env.valid_actions(), done)
    assert agent.weights
    assert all(isinstance(table, np.ndarray) for table in agent.weights.values())
    assert all(table.shape == (agent.config.expansion_width,) for table in agent.weights.values())


def test_eligibility_traces_use_dense_numpy_vectors():
    import numpy as np

    agent = SparseFlyAgent(seed=5)
    env = ASTMaze(TASKS["sum_list"])
    obs = env.observe()
    action, features = agent.choose(obs, env.valid_actions())
    next_obs, reward, done, _ = env.step(action)
    agent.learn(features, action, reward, next_obs, env.valid_actions(), done)
    assert agent.traces
    assert agent.memory_traces
    assert all(isinstance(v, np.ndarray) for v in agent.traces.values())
    assert all(isinstance(v, np.ndarray) for v in agent.memory_traces.values())


def test_batched_fast_q_matches_scalar_fast_q():
    agent = SparseFlyAgent(seed=8)
    env = MotorMaze(TASKS["count_positive"])
    obs = env.observe()
    features = agent.features(obs)
    actions = env.valid_actions()

    # Seed a few component tables so the test covers non-zero values.
    for action in actions:
        for key in action_components(action):
            table = agent._fast_table(key)
            for index in features[:3]:
                table[index] += 0.125

    batched = agent.q_fast_many(actions, features)
    for action in actions:
        assert abs(batched[action] - agent.q_fast(action, features)) < 1e-12


def test_sparse_encoder_reuses_token_indices():
    enc = SparseEncoder(width=1024, hashes_per_token=3)
    first = enc._indices("sense:test")
    second = enc._indices("sense:test")
    assert first is second
    assert len(enc._index_cache) == 1


def test_role_components_are_not_in_broad_memory_channel():
    assert "control:init" not in memory_action_components("SET:acc=0")
    assert "control:unconditional" not in memory_action_components("AUG:acc+=1")
    assert "control:init" in factor_memory_components("SET:acc=0")
    assert "control:unconditional" in factor_memory_components("AUG:acc+=1")


def test_role_factor_routing_is_semantic():
    assert concept_accepts_factor("role:identity_zero", "control:init")
    assert concept_accepts_factor("role:identity_zero", "effect_rhs:0")
    assert not concept_accepts_factor("role:identity_zero", "effect_rhs:x")
    assert concept_accepts_factor("role:list_reduce", "control:loop")
    assert concept_accepts_factor("role:list_reduce", "control:return")


def test_role_mix_zero_cannot_change_q():
    agent = SparseFlyAgent(
        FlyConfig(slow_mix=0.45, role_factor_mix=0.0),
        seed=9,
    )
    env = MotorMaze(TASKS["sum_list"])
    obs = env.observe()
    features = agent.features(obs)
    memory = agent.memory_features(obs)
    action = env.valid_actions()[0]
    agent._active_context = "none"
    agent._active_concepts = ("role:identity_zero",)
    agent.factor_concept_weights["role:identity_zero"] = {
        "control:init": {memory[0]: 100.0},
        "effect_rhs:0": {memory[0]: 100.0},
    }
    assert agent.q(action, features, memory) == agent.q_fast(action, features)


def test_factor_channels_are_gated_independently():
    config = FlyConfig(
        slow_mix=0.45,
        role_factor_mix=0.30,
        agg_factor_mix=0.15,
        filter_factor_mix=0.20,
    )
    assert config.role_factor_mix == 0.30
    assert config.agg_factor_mix == 0.15
    assert config.filter_factor_mix == 0.20
    assert concept_accepts_factor("agg:count", "effect_rhs:1")
    assert concept_accepts_factor("filter:all", "control:unconditional")
