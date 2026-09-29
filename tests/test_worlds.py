from mosca.benchmark import (
    _learn_motor_episode,
    cross_world_transfer_suite,
    transfer_benchmark,
)
from mosca.fly import FlyConfig, SparseFlyAgent
from mosca.mcts import mcts_solve
from mosca.motor_env import MotorMaze, OracleCounter
from mosca.native_world import NativeWorld
from mosca.typed_world import TypedWorld
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


def test_native_mosca_world_is_registered_and_independent():
    assert "mosca" in world_names()
    world = create_world("mosca", TASKS["sum_list"])
    assert isinstance(world, ProgramWorld)
    assert isinstance(world, NativeWorld)
    assert not isinstance(world, MotorMaze)


def _step_pair(task_name: str, actions: tuple[str, ...]):
    python = create_world("python", TASKS[task_name])
    native = create_world("mosca", TASKS[task_name])
    for action in actions:
        assert python.observe() == native.observe()
        assert python.valid_actions() == native.valid_actions()
        p_obs, p_reward, p_done, p_info = python.step(action)
        n_obs, n_reward, n_done, n_info = native.step(action)
        assert p_obs == n_obs
        assert abs(p_reward - n_reward) < 1e-12
        assert p_done == n_done
        assert p_info.get("probe_before") == n_info.get("probe_before")
        assert p_info.get("probe_after") == n_info.get("probe_after")
    return python, native


def test_native_world_matches_reference_program_semantics():
    programs = {
        "sum_list": (
            "SET:acc=0", "FOR:x:xs", "AUG:acc+=x", "END", "RETURN:acc",
        ),
        "count_positive": (
            "SET:acc=0", "FOR:x:xs", "WHEN:x>0:INC1", "END", "RETURN:acc",
        ),
        "max_list": (
            "SET:acc=xs[0]", "FOR:x:xs", "WHEN:x>acc:SETX", "END", "RETURN:acc",
        ),
    }
    for task_name, actions in programs.items():
        python, native = _step_pair(task_name, actions)
        assert python.last_evaluation is not None
        assert native.last_evaluation is not None
        assert python.last_evaluation.score == native.last_evaluation.score == 1.0
        assert python.evaluate_hidden().score == native.evaluate_hidden().score == 1.0
        assert python.counter.snapshot() == native.counter.snapshot()


def test_native_world_matches_random_valid_trajectories():
    import random

    for task_name in ("sum_list", "count_positive", "max_list"):
        for seed in range(8):
            rng = random.Random(seed)
            python = create_world("python", TASKS[task_name])
            native = create_world("mosca", TASKS[task_name])
            while not python.done and len(python.actions) < 8:
                assert python.observe() == native.observe()
                valid = python.valid_actions()
                assert valid == native.valid_actions()
                if not valid:
                    break
                action = rng.choice(valid)
                p_obs, p_reward, p_done, _ = python.step(action)
                n_obs, n_reward, n_done, _ = native.step(action)
                assert p_obs == n_obs
                assert abs(p_reward - n_reward) < 1e-12
                assert p_done == n_done
            if python.last_evaluation is not None:
                assert native.last_evaluation is not None
                assert python.last_evaluation.score == native.last_evaluation.score
            assert python.counter.snapshot() == native.counter.snapshot()


def test_native_world_learning_episode_matches_python():
    config = FlyConfig(
        epsilon=0.30,
        alpha=0.065,
        gamma=0.98,
        trace_decay=0.93,
        history=8,
    )
    python_agent = SparseFlyAgent(config, seed=73)
    native_agent = SparseFlyAgent(config, seed=73)
    python_counter = OracleCounter()
    native_counter = OracleCounter()

    python_result = _learn_motor_episode(
        python_agent,
        TASKS["count_positive"],
        counter=python_counter,
        world_name="python",
    )
    native_result = _learn_motor_episode(
        native_agent,
        TASKS["count_positive"],
        counter=native_counter,
        world_name="mosca",
    )
    assert python_result == native_result
    assert python_counter.snapshot() == native_counter.snapshot()
    assert python_agent.parameter_count() == native_agent.parameter_count()


def test_native_world_mcts_matches_python_search_path():
    python = mcts_solve(
        TASKS["sum_list"],
        simulations=80,
        seed=31,
        world_name="python",
    )
    native = mcts_solve(
        TASKS["sum_list"],
        simulations=80,
        seed=31,
        world_name="mosca",
    )
    assert python.first_visible_at == native.first_visible_at
    assert python.first_generalizing_at == native.first_generalizing_at
    assert python.best_train_score == native.best_train_score
    assert python.best_hidden_score == native.best_hidden_score
    assert python.actions == native.actions
    assert (
        python.visible_oracle_calls_at_first_generalizing
        == native.visible_oracle_calls_at_first_generalizing
    )
    assert python.source != native.source


def test_typed_world_is_registered_and_independent():
    assert "typed" in world_names()
    world = create_world("typed", TASKS["sum_list"])
    assert isinstance(world, TypedWorld)
    assert isinstance(world, NativeWorld)


def test_typed_world_reduces_loop_branching_without_extra_depth():
    python = create_world("python", TASKS["count_positive"])
    typed = create_world("typed", TASKS["count_positive"])

    python.step("SET:acc=0")
    python.step("FOR:x:xs")
    typed.step("SET:acc=0")
    typed.step("FOR:x:xs")

    assert len(python.valid_actions()) == 21
    assert len(typed.valid_actions()) == 12

    actions = (
        "SET:acc=0",
        "FOR:x:xs",
        "WHEN:x>0:INC1",
        "END",
        "RETURN:acc",
    )
    typed = create_world("typed", TASKS["count_positive"])
    for action in actions:
        typed.step(action)
    assert len(actions) == 5
    assert typed.last_evaluation is not None
    assert typed.last_evaluation.score == 1.0


def test_typed_world_keeps_all_current_reference_semantics():
    programs = {
        "sum_list": (
            "SET:acc=0", "FOR:x:xs", "AUG:acc+=x", "END", "RETURN:acc",
        ),
        "count_positive": (
            "SET:acc=0", "FOR:x:xs", "WHEN:x>0:INC1", "END", "RETURN:acc",
        ),
        "max_list": (
            "SET:acc=xs[0]", "FOR:x:xs", "WHEN:x>acc:SETX", "END", "RETURN:acc",
        ),
    }
    for name, actions in programs.items():
        world = create_world("typed", TASKS[name])
        for action in actions:
            world.step(action)
        assert world.last_evaluation is not None
        assert world.last_evaluation.score == 1.0
        assert world.evaluate_hidden().score == 1.0


def test_typed_world_excludes_role_mismatched_conditionals():
    world = create_world("typed", TASKS["count_positive"])
    world.step("SET:acc=0")
    world.step("FOR:x:xs")
    valid = set(world.valid_actions())

    assert "WHEN:x>0:INC1" in valid
    assert "WHEN:x<0:ADDX" in valid
    assert "WHEN:x>acc:SETX" in valid

    assert "WHEN:x>acc:INC1" not in valid
    assert "WHEN:x>0:SETX" not in valid
    assert "WHEN:x<acc:ADDX" not in valid


def test_cross_world_transfer_reports_source_and_target_worlds():
    result = cross_world_transfer_suite(
        pretrain_episodes=6,
        adapt_episodes=3,
        seeds=(0,),
        source_world="python",
        target_world="typed",
    )
    assert result["source_world"] == "python"
    assert result["target_world"] == "typed"
    assert set(result["summary"])


def test_cross_world_same_world_matches_transfer_semantics_smoke():
    direct = transfer_benchmark(
        pretrain_episodes=6,
        adapt_episodes=3,
        seed=4,
        target_name="sum_positive",
        world_name="typed",
    )
    cross = cross_world_transfer_suite(
        pretrain_episodes=6,
        adapt_episodes=3,
        seeds=(4,),
        source_world="typed",
        target_world="typed",
    )
    target = cross["runs"][0]["targets"]["sum_positive"]
    assert direct["transfer"]["first_generalized"] == target["transfer"]["first_generalized"]
    assert direct["transfer"]["generalized"] == target["transfer"]["generalized"]
