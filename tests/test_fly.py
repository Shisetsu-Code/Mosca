from mosca.ast_env import ASTMaze
from mosca.fly import FlyConfig, SparseEncoder, SparseFlyAgent, action_components
from mosca.motor_env import MotorMaze
from mosca.tasks import TASKS, TRANSFER_TASKS


def test_sparse_encoder_is_deterministic_and_sparse():
    obs = MotorMaze(TASKS["sum_list"]).observe()
    enc = SparseEncoder(width=1024, hashes_per_token=3)
    a = enc.encode(obs)
    b = enc.encode(obs)
    assert a == b
    assert 0 < len(a) < 300


def test_task_name_is_not_in_motor_observation():
    obs = MotorMaze(TASKS["sum_list"]).observe()
    assert "task" not in obs
    assert obs["sensory"]


def test_action_factorization_exposes_reusable_parts():
    parts = action_components("WHEN:x>0:ADDX")
    assert {"op:WHEN", "cmp:>", "cond_rhs:0", "update:ADDX"} <= set(parts)


def test_fly_agent_updates_factor_weights():
    env = ASTMaze(TASKS["sum_list"])
    agent = SparseFlyAgent(seed=1)
    agent.begin_episode()
    obs = env.observe()
    action, features = agent.choose(obs, env.valid_actions())
    next_obs, reward, done, _ = env.step(action)
    agent.learn(features, action, reward, next_obs, env.valid_actions(), done)
    assert agent.parameter_count() > 0


def test_success_can_consolidate_into_concept_memory():
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
    assert any(key == "concept:agg:sum" for key in agent.slow_weights)


def test_composed_task_reuses_two_source_memory_keys():
    sum_keys = set(SparseFlyAgent.memory_keys(MotorMaze(TASKS["sum_list"]).observe()))
    count_keys = set(SparseFlyAgent.memory_keys(MotorMaze(TASKS["count_positive"]).observe()))
    target_keys = set(SparseFlyAgent.memory_keys(MotorMaze(TRANSFER_TASKS["sum_positive"]).observe()))
    assert "concept:agg:sum" in sum_keys & target_keys
    assert "concept:filter:positive" in count_keys & target_keys
