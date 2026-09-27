from mosca.motor_env import MotorMaze, OracleCounter
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
        assert env.evaluate_hidden().score == 1.0, name
        assert len(env.actions) == 5, name


def test_goal_inhibition_closes_a_solved_program():
    env = MotorMaze(TASKS["count_positive"], max_steps=8)
    for action in ("SET:acc=0", "FOR:x:xs", "WHEN:x>0:INC1"):
        env.step(action)
    assert env.probe_score() == 1.0
    assert env.valid_actions() == ("END",)
    env.step("END")
    assert env.valid_actions() == ("RETURN:acc",)



def test_oracle_counter_separates_visible_and_hidden_work():
    counter = OracleCounter()
    env = MotorMaze(TASKS["sum_list"], max_steps=8, counter=counter)
    for action in MOTOR_REFERENCE_ACTIONS["sum_list"]:
        env.step(action)

    visible = counter.snapshot()
    assert visible["visible_oracle_calls"] > 0
    assert visible["visible_case_executions"] > 0
    assert visible["hidden_oracle_calls"] == 0
    assert visible["hidden_case_executions"] == 0

    env.evaluate_hidden()
    hidden = counter.snapshot()
    assert hidden["hidden_oracle_calls"] == 1
    assert hidden["hidden_case_executions"] == len(TASKS["sum_list"].hidden_cases)
    assert hidden["visible_oracle_calls"] == visible["visible_oracle_calls"]
