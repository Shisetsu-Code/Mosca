from mosca.ast_env import ASTMaze
from mosca.fly import SparseEncoder, SparseFlyAgent, action_components
from mosca.motor_env import MotorMaze
from mosca.tasks import TASKS


def test_sparse_encoder_is_deterministic_and_sparse():
    obs = MotorMaze(TASKS["sum_list"]).observe()
    enc = SparseEncoder(width=1024, hashes_per_token=3)
    a = enc.encode(obs)
    b = enc.encode(obs)
    assert a == b
    assert 0 < len(a) < 300
    assert all(0 <= i < 1024 for i in a)


def test_task_name_is_not_in_motor_observation():
    obs = MotorMaze(TASKS["sum_list"]).observe()
    assert "task" not in obs
    assert obs["sensory"]
    assert all("sum_list" not in token for token in obs["sensory"])


def test_action_factorization_exposes_reusable_parts():
    parts = action_components("WHEN:x>0:ADDX")
    assert "op:WHEN" in parts
    assert "cmp:>" in parts
    assert "cond_rhs:0" in parts
    assert "update:ADDX" in parts


def test_fly_agent_updates_factor_weights():
    env = ASTMaze(TASKS["sum_list"])
    agent = SparseFlyAgent(seed=1)
    agent.begin_episode()
    obs = env.observe()
    action, features = agent.choose(obs, env.valid_actions())
    next_obs, reward, done, _ = env.step(action)
    agent.learn(features, action, reward, next_obs, env.valid_actions(), done)
    assert agent.parameter_count() > 0



def test_success_can_consolidate_fast_weights_into_slow_weights():
    from mosca.fly import FlyConfig

    config = FlyConfig(
        consolidation_rate=1.0,
        slow_mix=0.5,
        consolidation_threshold=-1.0,
    )
    agent = SparseFlyAgent(config, seed=3)
    env = MotorMaze(TASKS["sum_list"])
    agent.begin_episode()
    obs = env.observe()
    action, features = agent.choose(obs, env.valid_actions())
    next_obs, reward, _, _ = env.step(action)
    agent.learn(features, action, max(reward, 0.0), next_obs, env.valid_actions(), True)
    assert agent.fast_parameter_count() > 0
    assert agent.slow_parameter_count() > 0
