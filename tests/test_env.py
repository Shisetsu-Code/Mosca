from mosca.env import Action, PythonMaze
from mosca.tasks import TASKS


def run(task_name: str, actions: list[Action]):
    env = PythonMaze(TASKS[task_name], max_steps=20)
    for action in actions:
        _, _, done, _ = env.step(action)
    assert done
    assert env.last_evaluation is not None
    return env


def test_sum_program():
    env = run("sum_list", [
        Action.ACC_ZERO,
        Action.FOR_X_XS,
        Action.ACC_ADD_X,
        Action.END_BLOCK,
        Action.RETURN_ACC,
        Action.COMMIT,
    ])
    assert env.last_evaluation.score == 1.0


def test_count_positive_program():
    env = run("count_positive", [
        Action.ACC_ZERO,
        Action.FOR_X_XS,
        Action.IF_X_GT_ZERO,
        Action.ACC_ADD_ONE,
        Action.END_BLOCK,
        Action.END_BLOCK,
        Action.RETURN_ACC,
        Action.COMMIT,
    ])
    assert env.last_evaluation.score == 1.0


def test_max_program():
    env = run("max_list", [
        Action.ACC_FIRST,
        Action.FOR_X_XS,
        Action.IF_X_GT_ACC,
        Action.ACC_SET_X,
        Action.END_BLOCK,
        Action.END_BLOCK,
        Action.RETURN_ACC,
        Action.COMMIT,
    ])
    assert env.last_evaluation.score == 1.0
