from mosca.search import bfs_solve
from mosca.tasks import TASKS


def test_bfs_finds_reference_tasks():
    for name, depth in (("sum_list", 7), ("count_positive", 9), ("max_list", 9)):
        result = bfs_solve(TASKS[name], max_depth=depth)
        assert result is not None, name
        assert result.score == 1.0, name
