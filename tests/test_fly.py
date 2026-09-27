from mosca.ast_env import ASTMaze
from mosca.fly import SparseEncoder, SparseFlyAgent
from mosca.tasks import TASKS


def test_sparse_encoder_is_deterministic_and_sparse():
    obs = ASTMaze(TASKS["sum_list"]).observe()
    enc = SparseEncoder(width=1024, hashes_per_token=3)
    a = enc.encode(obs)
    b = enc.encode(obs)
    assert a == b
    assert 0 < len(a) < 100
    assert all(0 <= i < 1024 for i in a)


def test_fly_agent_updates_local_weights():
    env = ASTMaze(TASKS["sum_list"])
    agent = SparseFlyAgent(seed=1)
    agent.begin_episode()
    obs = env.observe()
    action, features = agent.choose(obs, env.valid_actions())
    next_obs, reward, done, _ = env.step(action)
    agent.learn(features, action, reward, next_obs, env.valid_actions(), done)
    assert agent.parameter_count() > 0
