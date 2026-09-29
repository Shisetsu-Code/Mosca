from mosca.benchmark import _learn_motor_episode, transfer_benchmark
from mosca.fly import FlyConfig, SparseFlyAgent
from mosca.mcts import mcts_solve
from mosca.motor_env import MotorMaze, OracleCounter
from mosca.compact_world import CompactWorld
from mosca.native_world import NativeWorld
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


def test_compact_world_is_registered():
    assert "compact" in world_names()
    world = create_world("compact", TASKS["count_positive"])
    assert isinstance(world, ProgramWorld)
    assert isinstance(world, CompactWorld)


COMPACT_PROGRAMS = {
    "sum_list": (
        "INIT:0", "EACH", "ADDX", "END", "YIELD",
    ),
    "count_positive": (
        "INIT:0", "EACH", "GUARD", "CMP:>", "RHS:0",
        "DO:INC1", "END", "YIELD",
    ),
    "max_list": (
        "INIT:first", "EACH", "GUARD", "CMP:>", "RHS:acc",
        "DO:SETX", "END", "YIELD",
    ),
}


def test_compact_reference_programs_match_python_semantics():
    for task_name, actions in COMPACT_PROGRAMS.items():
        world = create_world("compact", TASKS[task_name])
        for action in actions:
            world.step(action)
        assert world.done
        assert world.last_evaluation is not None
        assert world.last_evaluation.score == 1.0
        assert world.evaluate_hidden().score == 1.0


def test_compact_language_reduces_peak_branching():
    python = create_world("python", TASKS["count_positive"])
    compact = create_world("compact", TASKS["count_positive"])

    python_actions = (
        "SET:acc=0", "FOR:x:xs", "WHEN:x>0:INC1", "END", "RETURN:acc",
    )
    compact_actions = COMPACT_PROGRAMS["count_positive"]

    def branch_profile(world, actions):
        profile = []
        for action in actions:
            profile.append(len(world.valid_actions()))
            world.step(action)
        return profile

    python_profile = branch_profile(python, python_actions)
    compact_profile = branch_profile(compact, compact_actions)

    assert max(compact_profile) <= 5
    assert max(compact_profile) < max(python_profile)
    assert len(compact_actions) > len(python_actions)


def test_compact_action_semantics_share_canonical_components():
    from mosca.fly import (
        action_components,
        factor_memory_components,
        memory_action_components,
    )

    assert "op:FOR" in action_components("EACH")
    assert "cmp:>" in action_components("CMP:>")
    assert "cond_rhs:0" in memory_action_components("RHS:0")
    assert "effect:add" in memory_action_components("DO:ADDX")
    assert "control:init" in factor_memory_components("INIT:0")
    assert "control:conditional" in factor_memory_components("GUARD")


def test_compact_mcts_can_find_sum_program():
    result = mcts_solve(
        TASKS["sum_list"],
        simulations=120,
        seed=17,
        world_name="compact",
    )
    assert result.first_generalizing_at is not None
    assert result.best_hidden_score == 1.0
