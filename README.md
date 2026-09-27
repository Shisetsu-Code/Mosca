# Mosca

Experimental program-synthesis environment inspired by the navigation and reward-learning principles of *Drosophila*, without pretending that a fly connectome is already a code generator.

The core hypothesis is simple: treat programming as navigation through a constrained state space instead of next-token prediction.

## v0 model

- **World:** a deliberately tiny subset of Python.
- **Position:** current partial program / block context.
- **Moves:** valid structural actions (`FOR_X_XS`, `IF_X_GT_ZERO`, `ACC_ADD_X`, ...).
- **Physics:** CPython 3.12.14 `compile()` and execution semantics.
- **Reward:** compilation + fraction of tests passed.
- **Baseline:** breadth-first search, so the environment can be validated before introducing a neural agent.

The agent never emits arbitrary Python text. The environment owns the grammar and renders source from legal actions. That keeps syntax knowledge out of the model and makes the search space measurable.

## Pinned runtime

Mosca targets **CPython 3.12.14**. `.python-version` and `Dockerfile` pin it. Benchmark runs should use the pinned runtime.

## Run

```bash
python -m pip install -e .[dev]
pytest -q
mosca runtime
mosca solve sum_list --depth 7
mosca solve count_positive --depth 9
mosca solve max_list --depth 9
```

Or with Docker:

```bash
docker build -t mosca .
docker run --rm mosca
```

## Current action vocabulary

```text
ACC_ZERO       acc = 0
ACC_FIRST      acc = xs[0]
FOR_X_XS       for x in xs:
IF_X_GT_ACC    if x > acc:
IF_X_GT_ZERO   if x > 0:
ACC_ADD_X      acc += x
ACC_ADD_ONE    acc += 1
ACC_SET_X      acc = x
END_BLOCK      leave current block
RETURN_ACC     return acc
COMMIT         compile + run tests
```

This is intentionally tiny. v0 validates the navigation formulation; it is not intended to provide broad Python coverage.

## Roadmap

1. Validate the maze/oracle against deterministic search.
2. Replace handcrafted statements with a typed AST-hole grammar derived from CPython's AST/grammar definitions.
3. Add a sparse state encoder analogous to expansion coding in the mushroom body.
4. Add recurrent navigation state analogous in function, not literal anatomy, to central-complex navigation.
5. Add reward-modulated eligibility traces from compiler/test feedback.
6. Compare random search, BFS/MCTS, GRU, small Transformer and the sparse recurrent agent on identical tasks.
7. Only after the synthetic architecture is measurable, map selected FlyWire motifs into the network topology.

## Design rule

Do not train the agent on Python documentation unless an experiment specifically tests that condition. Language syntax and structural validity belong to the environment. The agent should learn which legal transformations solve a task.
