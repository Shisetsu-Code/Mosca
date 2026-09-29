from mosca.benchmark import _learn_motor_episode, transfer_benchmark
from mosca.fly import FlyConfig, SparseFlyAgent
from mosca.mcts import mcts_solve
from mosca.motor_env import MotorMaze, OracleCounter
from mosca.resource import resource_benchmark
from mosca.tasks import TASKS
from mosca.worlds import ProgramWorld, create_world, register_world, world_names


def _clone_factory(task, max_steps, counter):
    return MotorMaze(task, max_steps=max_steps, counter=counter)


def _register_clone():
    register_world("testlang", _clone_factory, replace=True)


def test_python_world_is_registered_and_contract_compatible():
    assert "python" in world_names()
    world = create_world("python", TASKS["sum_list"])
    assert isinstance(world, ProgramWorld)
    assert isinstance(world, MotorMaze)
    assert world.observe()["valid_actions"]


def test_world_registry_accepts_an_independent_backend():
    _register_clone()
    world = create_world("testlang", TASKS["sum_list"])
    assert isinstance(world, ProgramWorld)
    assert world.task.name == "sum_list"


def test_unknown_world_fails_explicitly():
    try:
        create_world("does-not-exist", TASKS["sum_list"])
    except ValueError as exc:
        assert "available=" in str(exc)
    else:
        raise AssertionError("unknown world should fail")


def test_clone_world_matches_python_learning_episode():
    _register_clone()
    config = FlyConfig(
        epsilon=0.30,
        alpha=0.065,
        gamma=0.98,
        trace_decay=0.93,
        history=8,
    )
    python_agent = SparseFlyAgent(config, seed=41)
    clone_agent = SparseFlyAgent(config, seed=41)
    python_counter = OracleCounter()
    clone_counter = OracleCounter()

    python_result = _learn_motor_episode(
        python_agent,
        TASKS["sum_list"],
        counter=python_counter,
        world_name="python",
    )
    clone_result = _learn_motor_episode(
        clone_agent,
        TASKS["sum_list"],
        counter=clone_counter,
        world_name="testlang",
    )

    assert python_result == clone_result
    assert python_counter.snapshot() == clone_counter.snapshot()
    assert python_agent.parameter_count() == clone_agent.parameter_count()


def test_clone_world_matches_python_mcts():
    _register_clone()
    python_result = mcts_solve(
        TASKS["sum_list"],
        simulations=40,
        seed=9,
        world_name="python",
    )
    clone_result = mcts_solve(
        TASKS["sum_list"],
        simulations=40,
        seed=9,
        world_name="testlang",
    )
    assert python_result == clone_result


def test_transfer_benchmark_reports_selected_world():
    _register_clone()
    result = transfer_benchmark(
        pretrain_episodes=6,
        adapt_episodes=3,
        seed=3,
        world_name="testlang",
    )
    assert result["world"] == "testlang"
    assert result["transfer"]["world"] == "testlang"
    assert result["scratch"]["world"] == "testlang"


def test_resource_benchmark_accepts_registered_world():
    _register_clone()
    result = resource_benchmark(
        seed=2,
        pretrain_episodes=3,
        adapt_episodes=3,
        mcts_simulations=3,
        random_episodes=3,
        world_name="testlang",
    )
    assert result["world"] == "testlang"
    assert result["transfer"]["world"] == "testlang"
    assert result["mcts"]["resources"]["cpu_seconds"] >= 0
