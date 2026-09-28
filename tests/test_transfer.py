from mosca.benchmark import transfer_benchmark, transfer_suite
from mosca.motor_env import MotorMaze
from mosca.sensory import task_memory_concepts
from mosca.tasks import TASKS, TRANSFER_TASKS


REFERENCE_ACTIONS = {
    "sum_positive": (
        "SET:acc=0", "FOR:x:xs", "WHEN:x>0:ADDX", "END", "RETURN:acc",
    ),
    "count_all": (
        "SET:acc=0", "FOR:x:xs", "AUG:acc+=1", "END", "RETURN:acc",
    ),
    "max_positive_or_zero": (
        "SET:acc=0", "FOR:x:xs", "WHEN:x>acc:SETX", "END", "RETURN:acc",
    ),
}


def test_all_transfer_tasks_are_expressible_and_hidden():
    for name, task in TRANSFER_TASKS.items():
        assert task.hidden_cases, name
        env = MotorMaze(task, max_steps=8)
        for action in REFERENCE_ACTIONS[name]:
            env.step(action)
        assert env.last_evaluation is not None, name
        assert env.last_evaluation.score == 1.0, name
        assert env.evaluate_hidden().score == 1.0, name


def test_memory_concepts_compose_source_behaviors():
    assert "agg:sum" in task_memory_concepts(TASKS["sum_list"])
    assert "agg:count" in task_memory_concepts(TASKS["count_positive"])
    assert "agg:max" in task_memory_concepts(TASKS["max_list"])
    assert "filter:all" in task_memory_concepts(TASKS["sum_list"])
    assert "filter:positive" in task_memory_concepts(TASKS["count_positive"])

    expected = {
        "sum_positive": {"agg:sum", "filter:positive"},
        "count_all": {"agg:count", "filter:all"},
        "max_positive_or_zero": {"agg:max", "filter:positive"},
    }
    for name, concepts in expected.items():
        assert concepts <= set(task_memory_concepts(TRANSFER_TASKS[name]))


def test_transfer_benchmark_accepts_each_target():
    for name in TRANSFER_TASKS:
        result = transfer_benchmark(
            pretrain_episodes=6,
            adapt_episodes=3,
            seed=7,
            target_name=name,
        )
        assert result["target"] == name
        assert "transfer_zero_shot" in result
        assert result["transfer"]["episodes"] == 3


def test_transfer_suite_smoke():
    result = transfer_suite(
        pretrain_episodes=6,
        adapt_episodes=3,
        seeds=(0,),
    )
    assert set(result["targets"]) == set(TRANSFER_TASKS)
    assert set(result["summary"]) == set(TRANSFER_TASKS)
    assert len(result["runs"]) == 1
