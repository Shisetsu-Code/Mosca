from mosca.ast_env import ASTMaze
from mosca.grammar import rule_manifest, validate_runtime_schema
from mosca.reference import REFERENCE_ACTIONS
from mosca.tasks import SOURCE_TASKS


def test_runtime_ast_schema_is_compatible():
    fields = validate_runtime_schema()
    assert "For" in fields
    assert rule_manifest()["production_count"] == 28


def test_generic_ast_reference_programs():
    for name, actions in REFERENCE_ACTIONS.items():
        env = ASTMaze(SOURCE_TASKS[name], max_steps=50)
        for action in actions:
            env.step(action)
        assert env.done, name
        assert env.last_evaluation is not None, name
        assert env.last_evaluation.score == 1.0, name
        assert env.evaluate_hidden().score == 1.0, name
        src = env.source()
        assert src and "def solve(xs):" in src
