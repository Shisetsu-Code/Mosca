from mosca.resource import resource_benchmark


def test_resource_benchmark_smoke():
    result = resource_benchmark(
        seed=1,
        pretrain_episodes=3,
        adapt_episodes=3,
        mcts_simulations=3,
        random_episodes=3,
    )
    assert result["target"] == "sum_positive"
    for key in ("transfer", "scratch", "mcts", "random"):
        resources = result[key]["resources"]
        assert resources["wall_seconds"] >= 0
        assert resources["cpu_seconds"] >= 0
        assert resources["peak_tracemalloc_bytes"] > 0
