from mosca.benchmark import (
    DEFAULT_TRANSFER_AGG_FACTOR_MIX,
    DEFAULT_TRANSFER_FILTER_FACTOR_MIX,
    DEFAULT_TRANSFER_ROLE_FACTOR_MIX,
    transfer_benchmark,
    transfer_suite,
)
from mosca.motor_env import MotorMaze
from mosca.sensory import task_memory_concepts
from mosca.tasks import SOURCE_TASKS, TASKS, TRANSFER_TASKS


REFERENCE_ACTIONS = {
    "sum_positive": (
        "SET:acc=0", "FOR:x:xs", "WHEN:x>0:ADDX", "END", "RETURN:acc",
    ),
    "sum_negative": (
        "SET:acc=0", "FOR:x:xs", "WHEN:x<0:ADDX", "END", "RETURN:acc",
    ),
    "count_all": (
        "SET:acc=0", "FOR:x:xs", "AUG:acc+=1", "END", "RETURN:acc",
    ),
    "max_positive_or_zero": (
        "SET:acc=0", "FOR:x:xs", "WHEN:x>acc:SETX", "END", "RETURN:acc",
    ),
    "min_negative_or_zero": (
        "SET:acc=0", "FOR:x:xs", "WHEN:x<acc:SETX", "END", "RETURN:acc",
    ),
}


def test_all_transfer_tasks_are_expressible_and_hidden():
    assert set(REFERENCE_ACTIONS) == set(TRANSFER_TASKS)
    for name, task in TRANSFER_TASKS.items():
        assert task.hidden_cases, name
        env = MotorMaze(task, max_steps=8)
        for action in REFERENCE_ACTIONS[name]:
            env.step(action)
        assert env.last_evaluation is not None, name
        assert env.last_evaluation.score == 1.0, name
        assert env.evaluate_hidden().score == 1.0, name


def test_expanded_source_curriculum_contains_new_primitive_concepts():
    assert set(TASKS) < set(SOURCE_TASKS)
    assert {"count_negative", "min_list"} <= set(SOURCE_TASKS)
    assert "filter:negative" in task_memory_concepts(SOURCE_TASKS["count_negative"])
    assert "agg:min" in task_memory_concepts(SOURCE_TASKS["min_list"])


def test_memory_concepts_compose_source_behaviors():
    assert "agg:sum" in task_memory_concepts(SOURCE_TASKS["sum_list"])
    assert "agg:count" in task_memory_concepts(SOURCE_TASKS["count_positive"])
    assert "agg:max" in task_memory_concepts(SOURCE_TASKS["max_list"])
    assert "agg:min" in task_memory_concepts(SOURCE_TASKS["min_list"])
    assert "filter:all" in task_memory_concepts(SOURCE_TASKS["sum_list"])
    assert "filter:positive" in task_memory_concepts(SOURCE_TASKS["count_positive"])
    assert "filter:negative" in task_memory_concepts(SOURCE_TASKS["count_negative"])

    expected = {
        "sum_positive": {
            "agg:sum", "filter:positive",
            "role:identity_zero", "role:list_reduce",
        },
        "sum_negative": {
            "agg:sum", "filter:negative",
            "role:identity_zero", "role:list_reduce",
        },
        "count_all": {
            "agg:count", "filter:all",
            "role:identity_zero", "role:list_reduce",
        },
        "max_positive_or_zero": {
            "agg:max", "filter:positive",
            "role:identity_zero", "role:list_reduce",
        },
        "min_negative_or_zero": {
            "agg:min", "filter:negative",
            "role:identity_zero", "role:list_reduce",
        },
    }
    for name, concepts in expected.items():
        assert concepts <= set(task_memory_concepts(TRANSFER_TASKS[name]))


def test_transfer_benchmark_accepts_each_target():
    for name in TRANSFER_TASKS:
        result = transfer_benchmark(
            pretrain_episodes=10,
            adapt_episodes=3,
            seed=7,
            target_name=name,
            source_curriculum="expanded",
        )
        assert result["target"] == name
        assert result["source_curriculum"] == "expanded"
        assert "transfer_zero_shot" in result
        assert result["transfer"]["episodes"] == 3


def test_transfer_suite_smoke():
    result = transfer_suite(
        pretrain_episodes=10,
        adapt_episodes=3,
        seeds=(0,),
        source_curriculum="expanded",
    )
    assert set(result["targets"]) == set(TRANSFER_TASKS)
    assert set(result["summary"]) == set(TRANSFER_TASKS)
    assert result["source_curriculum"] == "expanded"
    assert len(result["runs"]) == 1


def test_core_curriculum_remains_available_as_control():
    result = transfer_suite(
        pretrain_episodes=6,
        adapt_episodes=2,
        seeds=(0,),
        source_curriculum="core",
    )
    assert result["source_curriculum"] == "core"


def test_role_factor_mix_is_reported_by_suite():
    result = transfer_suite(
        pretrain_episodes=10,
        adapt_episodes=2,
        seeds=(0,),
        role_factor_mix=0.25,
    )
    assert result["role_factor_mix"] == 0.25


def test_factor_channel_mixes_are_reported_by_suite():
    result = transfer_suite(
        pretrain_episodes=10,
        adapt_episodes=2,
        seeds=(0,),
        role_factor_mix=0.30,
        agg_factor_mix=0.15,
        filter_factor_mix=0.20,
    )
    assert result["role_factor_mix"] == 0.30
    assert result["agg_factor_mix"] == 0.15
    assert result["filter_factor_mix"] == 0.20


def test_validated_transfer_defaults_are_enabled():
    result = transfer_suite(
        pretrain_episodes=10,
        adapt_episodes=2,
        seeds=(0,),
    )
    assert result["role_factor_mix"] == DEFAULT_TRANSFER_ROLE_FACTOR_MIX == 0.30
    assert result["agg_factor_mix"] == DEFAULT_TRANSFER_AGG_FACTOR_MIX == 0.15
    assert result["filter_factor_mix"] == DEFAULT_TRANSFER_FILTER_FACTOR_MIX == 0.0
