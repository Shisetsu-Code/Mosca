from mosca.mcts import mcts_benchmark, mcts_solve
from mosca.tasks import TASKS


def test_mcts_result_is_reproducible():
    a = mcts_solve(TASKS["sum_list"], simulations=40, seed=7)
    b = mcts_solve(TASKS["sum_list"], simulations=40, seed=7)
    assert a == b
    assert 0.0 <= a.best_train_score <= 1.0
    assert 0.0 <= a.hidden_score <= 1.0


def test_mcts_benchmark_contains_core_and_transfer_tasks():
    result = mcts_benchmark(simulations=5, seed=3)
    assert "sum_list" in result["tasks"]
    assert "count_positive" in result["tasks"]
    assert "max_list" in result["tasks"]
    assert "sum_positive" in result["tasks"]
