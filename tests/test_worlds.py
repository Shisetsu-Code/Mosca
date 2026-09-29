from mosca.motor_env import MotorMaze
from mosca.tasks import TASKS
from mosca.worlds import ProgramWorld, create_world, register_world, world_names


def test_python_world_is_registered_and_contract_compatible():
    assert "python" in world_names()
    world = create_world("python", TASKS["sum_list"])
    assert isinstance(world, ProgramWorld)
    assert isinstance(world, MotorMaze)
    assert world.observe()["valid_actions"]


def test_world_registry_accepts_an_independent_backend():
    def clone_factory(task, max_steps, counter):
        return MotorMaze(task, max_steps=max_steps, counter=counter)

    register_world("testlang", clone_factory, replace=True)
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
