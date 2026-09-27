from mosca.mcts import mcts_benchmark, mcts_solve
from mosca.tasks import TASKS


def test_mcts_result_is_reproducible():
    a = mcts_solve(TASKS["sum_list"], simulations=40, seed=7)
    b = mcts_solve(TASKS["sum_list"], simulations=40, seed=7)
    assert a == b
    assert 0.0 <= a.best_train_score <= 1.0
    assert 0.0 <= a.best_hidden_score <= 1.0
    assert a.simulations == 40
    assert a.visible_oracle_calls > 0
    assert a.visible_case_executions > 0


def test_mcts_benchmark_contains_core_and_transfer_tasks():
    result = mcts_benchmark(simulations=5, seed=3)
    assert "sum_list" in result["tasks"]
    assert "count_positive" in result["tasks"]
    assert "max_list" in result["tasks"]
    assert "sum_positive" in result["tasks"]



def test_mcts_multiseed_shape():
    from mosca.mcts import mcts_multiseed

    result = mcts_multiseed(simulations=3, seeds=(0, 1))
    assert result["seeds"] == [0, 1]
    assert len(result["tasks"]["sum_list"]["runs"]) == 2



def test_hidden_does_not_stop_search():
    result = mcts_solve(TASKS["sum_list"], simulations=50, seed=0)
    assert result.simulations == 50



def test_mcts_reports_oracle_cost_at_generalizing_candidate():
    result = mcts_solve(TASKS["sum_list"], simulations=100, seed=0)
    if result.first_generalizing_at is not None:
        assert result.visible_oracle_calls_at_first_generalizing is not None
        assert result.visible_case_executions_at_first_generalizing is not None
        assert result.visible_oracle_calls_at_first_generalizing <= result.visible_oracle_calls
