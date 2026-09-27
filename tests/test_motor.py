from mosca.motor_env import MotorMaze
from mosca.reference import MOTOR_REFERENCE_ACTIONS
from mosca.tasks import TASKS


def test_motor_reference_programs_are_correct_and_short():
    for name, actions in MOTOR_REFERENCE_ACTIONS.items():
        env = MotorMaze(TASKS[name], max_steps=8)
        for action in actions:
            env.step(action)
        assert env.done, name
        assert env.last_evaluation is not None, name
        assert env.last_evaluation.score == 1.0, name
        assert len(env.actions) == 5, name


def test_goal_inhibition_closes_a_solved_program():
    env = MotorMaze(TASKS["count_positive"], max_steps=8)
    for action in ("SET:acc=0", "FOR:x:xs", "WHEN:x>0:INC1"):
        env.step(action)
    assert env.probe_score() == 1.0
    assert env.valid_actions() == ("END",)
    env.step("END")
    assert env.valid_actions() == ("RETURN:acc",)
