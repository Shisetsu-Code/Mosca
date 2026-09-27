# Mosca

Experimental program synthesis inspired by the navigation and reward-learning principles of *Drosophila*.

The working hypothesis is that programming can be represented as navigation through a constrained state space instead of next-token prediction. Python syntax and structural legality belong to the environment; the learner only selects legal transitions.

## Current architecture

```text
problem + tests
      ↓
typed AST with holes
      ↓
sparse state expansion
      ↓
recurrent action history
      ↓
action selection
      ↓
CPython 3.12.14 compile/execute
      ↓
test reward
      └────────────→ eligibility traces
```

Mosca does not currently simulate the full fly connectome. The first neural component is deliberately small: sparse expansion coding plus recurrent context and reward-modulated eligibility traces.

## Pinned Python world

The environment is fixed to **CPython 3.12.14**.

`mosca rules` inspects the actual `ast` classes in that runtime, verifies the structural fields used by Mosca and emits a deterministic rule-manifest fingerprint. The model is not trained on Python documentation.

The generic v1 grammar has 28 productions covering statement lists, assignment, augmented assignment, `for`, `if`, `return`, names, integer constants, arithmetic, comparisons and subscripting.

The action vocabulary is grammatical (`STMT:FOR`, `EXPR:COMPARE`, `CMPOP:GT`, ...), not prewritten source statements.

## Run

```bash
python -m pip install -e .[dev]
pytest -q
mosca runtime
mosca rules
mosca ast-reference all
mosca benchmark --episodes 200 --seed 42
```

## Two environments

`mosca.env.PythonMaze` is the original hand-shaped v0 environment and remains as a deterministic baseline.

`mosca.ast_env.ASTMaze` is the v1 environment. It starts with a typed `stmt_list` hole and repeatedly expands the first unresolved hole. A completed tree is converted to a real `ast.Module`, compiled by CPython and evaluated against tests.

Reference programs are stored only to verify that the generic grammar can express known solutions. They are not used by the from-scratch benchmark.

## Fly learner v0

`SparseFlyAgent` contains deterministic sparse expansion coding, short recurrent action history, action-local value weights, TD(lambda)-style eligibility traces and reward modulation from compiler/test results.

This is a functional analogue of a few useful insect-learning ideas, not a claim of biological equivalence.

## Current experiment

The repository compares random exploration, the legacy BFS baseline and sparse reward-learning on the generic AST maze.

The first generic tasks are `sum_list`, `count_positive`, and `max_list`. Their reference trajectories require 18-26 grammar decisions, making sparse terminal reward substantially harder than the original hand-shaped maze.

## Next milestones

1. Add curriculum or intermediate reward without leaking target source code.
2. Add macro-actions learned from repeated successful subtrees.
3. Generate larger hidden task suites procedurally.
4. Compare MCTS, GRU and a small Transformer on exactly the same grammar.
5. Replace synthetic sparse/recurrent connectivity with selected FlyWire-derived motifs only if controlled baselines justify it.
