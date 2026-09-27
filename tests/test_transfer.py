from mosca.benchmark import transfer_benchmark
from mosca.motor_env import MotorMaze
from mosca.sensory import task_sensory_tokens
from mosca.tasks import TASKS, TRANSFER_TASKS


def test_transfer_task_is_expressible_and_hidden_cases_are_separate():
    task = TRANSFER_TASKS["sum_positive"]
    assert task.hidden_cases
    env = MotorMaze(task, max_steps=8)
    for action in ("SET:acc=0", "FOR:x:xs", "WHEN:x>0:ADDX", "END", "RETURN:acc"):
        env.step(action)
    assert env.last_evaluation is not None
    assert env.last_evaluation.score == 1.0
    assert env.evaluate_hidden().score == 1.0


def test_transfer_benchmark_smoke():
    result = transfer_benchmark(pretrain_episodes=6, adapt_episodes=3, seed=7)
    assert result["target"] == "sum_positive"
    assert result["transfer"]["episodes"] == 3
    assert result["scratch"]["episodes"] == 3


def test_zero_shot_fields_are_reported():
    result = transfer_benchmark(pretrain_episodes=6, adapt_episodes=3, seed=11)
    assert "transfer_zero_shot" in result
    assert "scratch_zero_shot" in result
    assert result["transfer_zero_shot"]["episodes"] == 32


def test_sensory_concepts_factor_composition():
    sum_tokens = set(task_sensory_tokens(TASKS["sum_list"]))
    count_tokens = set(task_sensory_tokens(TASKS["count_positive"]))
    target_tokens = set(task_sensory_tokens(TRANSFER_TASKS["sum_positive"]))

    assert "concept:agg:sum" in sum_tokens
    assert "concept:filter:positive" in count_tokens
    assert "concept:agg:sum" in target_tokens
    assert "concept:filter:positive" in target_tokens
