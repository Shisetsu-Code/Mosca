# Mosca

Experimental program synthesis inspired by navigation and reward learning in *Drosophila*.

Mosca treats programming as navigation through a constrained state space. The learner never writes arbitrary Python text: CPython owns syntax/AST legality, while the learner selects legal structural or semantic actions.

## Runtime

The world is pinned to **CPython 3.12.14**. Python documentation is not used as model training data.

## Architecture

```text
visible I/O examples
      ↓
factorized sensory concepts (no task name)
      ↓
generic typed AST world
      ↓
semantic motor layer
      ↓
sparse expansion + recurrent history
      ↓
factorized action policy
      ↓
CPython compile/execute
      ↓
test reward + eligibility traces
```

The learner does not receive names such as `sum_list` or `max_list`. Its sensory input is derived from visible input/output examples. Stable I/O relations are decomposed into reusable axes such as:

```text
aggregation = sum
aggregation = count
aggregation = max

filter = all
filter = positive
filter = negative
```

Thus the unseen `sum_positive` task can share one sensory factor with `sum_list` and another with `count_positive`.

Motor actions are factorized too. For example:

```text
WHEN:x>0:ADDX
```

shares components such as `WHEN`, `x`, `>`, `0`, and `ADDX` with other actions.

## Environments

1. `PythonMaze`: original hand-shaped baseline.
2. `ASTMaze`: generic low-level grammar with typed holes.
3. `MotorMaze`: hierarchical list-reduction curriculum with generated statement-sized motor primitives.

The hierarchy matters. Low-level AST trajectories require roughly 18–26 decisions for the first tasks; the motor layer expresses the same solutions in about 5 semantic decisions.

## Hidden evaluation

Training reward uses only visible cases. Hidden cases are evaluated only after a complete training solution is found.

## Transfer experiment

Held-out compositional task:

```python
acc = 0
for x in xs:
    if x > 0:
        acc += x
return acc
```

This recombines behavior relevant to `sum_list` and `count_positive`.

With 5 deterministic seeds, 600 pretraining episodes over the three core tasks and 250 adaptation episodes on `sum_positive`:

| Metric | Pretrained | Scratch |
|---|---:|---:|
| Seeds finding a hidden-generalizing solution | 5/5 | 4/5 |
| Mean episode of first generalizing solution | 62.6 | 167.25 |
| Best first generalizing episode | 8 | 153 |

The paired comparison favored the pretrained agent in all 5 seeds for time-to-first-generalizing solution.

This is still a small engineering experiment, not a statistical claim about general program synthesis. Performance remains noisy: the number of later successful episodes is not uniformly better, and individual seeds can regress.

### Zero-shot

Zero-shot composition is **not demonstrated yet**. After pretraining, the agent did not solve `sum_positive` in 32 greedy rollouts for any of the 5 transfer seeds. The current benefit is faster adaptation, not immediate unseen-task synthesis.

## Run

```bash
python -m pip install -e .[dev]
pytest -q

mosca runtime
mosca rules
mosca motor-benchmark --episodes 300 --seed 42 --require-fly-solved --require-fly-generalized
mosca motor-multiseed --episodes 75 --seeds 0,1,2
mosca transfer-benchmark --pretrain-episodes 900 --adapt-episodes 300 --seed 42
mosca transfer-multiseed --pretrain-episodes 600 --adapt-episodes 250 --seeds 0,1,2,3,4
```

## Next milestones

1. Separate reusable long-term semantic memory from task-local value learning to target true zero-shot composition.
2. Add MCTS and GRU baselines over exactly the same motor action space.
3. Learn reusable motor options instead of predefining the `WHEN` option family.
4. Add several held-out compositions, not just `sum_positive`.
5. Expand beyond reductions: filters, maps, nested loops, multiple variables and functions.
6. Only after controlled baselines, test FlyWire-derived connectivity motifs against matched synthetic sparse networks.
