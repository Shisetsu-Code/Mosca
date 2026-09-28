from mosca.benchmark import transfer_benchmark
from mosca.motor_env import MotorMaze
from mosca.sensory import task_memory_concepts
from mosca.tasks import TASKS, TRANSFER_TASKS


def test_transfer_task_is_expressible_and_hidden_cases_are_separate():
    task = TRANSFER_TASKS["sum_positive"]
    env = MotorMaze(task, max_steps=8)
    for action in ("SET:acc=0", "FOR:x:xs", "WHEN:x>0:ADDX", "END", "RETURN:acc"):
        env.step(action)
    assert env.last_evaluation is not None
    assert env.last_evaluation.score == 1.0
    assert env.evaluate_hidden().score == 1.0


def test_memory_concepts_compose_source_behaviors():
    assert "agg:sum" in task_memory_concepts(TASKS["sum_list"])
    assert "filter:positive" in task_memory_concepts(TASKS["count_positive"])
    target = set(task_memory_concepts(TRANSFER_TASKS["sum_positive"]))
    assert {"agg:sum", "filter:positive"} <= target


def test_transfer_benchmark_reports_zero_shot():
    result = transfer_benchmark(pretrain_episodes=6, adapt_episodes=3, seed=7)
    assert result["target"] == "sum_positive"
    assert "transfer_zero_shot" in result
    assert result["transfer"]["episodes"] == 3


def test_default_concept_mix_preserves_legacy_slow_mix():
    from mosca.fly import FlyConfig, SparseFlyAgent

    base = SparseFlyAgent(FlyConfig(slow_mix=0.45, concept_mix=None), seed=1)
    explicit = SparseFlyAgent(FlyConfig(slow_mix=0.45, concept_mix=0.45), seed=1)
    env = MotorMaze(TASKS["sum_list"])
    obs = env.observe()
    features = base.features(obs)
    memory_features = base.memory_features(obs)
    action = env.valid_actions()[0]

    for agent in (base, explicit):
        agent._active_context = "none"
        agent._active_concepts = ("agg:sum",)
        agent.concept_weights["agg:sum"] = {
            "exact:" + action: {memory_features[0]: 1.0}
        }

    assert base.q(action, features, memory_features) == explicit.q(
        action, features, memory_features
    )


def test_concept_zero_shot_smoke():
    from mosca.benchmark import concept_zero_shot

    result = concept_zero_shot(pretrain_episodes=3, seed=2, concept_mix=0.75)
    assert result["concept_mix"] == 0.75
    assert result["zero_shot"]["episodes"] == 1
