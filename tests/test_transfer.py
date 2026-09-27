from mosca.benchmark import transfer_benchmark
from mosca.motor_env import MotorMaze
from mosca.tasks import TRANSFER_TASKS


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
