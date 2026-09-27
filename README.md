# Mosca

Experimental program synthesis inspired by navigation and reward learning in *Drosophila*.

Mosca treats programming as navigation through a constrained state space. The learner never writes arbitrary Python text: CPython owns syntax/AST legality, while the learner selects legal structural or semantic actions.

## Runtime

The world is pinned to **CPython 3.12.14**. Python documentation is not used as model training data.

## Architecture

```text
visible I/O examples
      ↓
fast sensory signature (no task name)
      ↓
sparse recurrent policy
      │
      ├── fast TD(lambda) weights
      ├── exact-context slow memory
      └── compositional concept memory
             ↓
       shared motor semantics
      ↓
semantic MotorMaze / AST
      ↓
CPython compile + execute
      ↓
visible-test reward
```

The fast policy keeps the stable v0.6 representation. Slow concept memory is deliberately separated: it uses a task-independent state encoding and semantic motor effects such as `add x`, `add 1`, and `set x`.

Stable I/O relations infer memory concepts such as:

```text
agg:sum
agg:count
agg:max
filter:all
filter:positive
filter:negative
```

Thus a new task can query memories acquired under different source tasks without inserting the task name into the network.

## Environments

1. `PythonMaze`: original hand-shaped baseline.
2. `ASTMaze`: generic low-level grammar with typed holes.
3. `MotorMaze`: hierarchical semantic action space.

The first benchmark solutions require about five motor decisions, versus roughly 18–26 low-level AST decisions.

## Hidden evaluation

Reward/probe uses only visible training cases. Hidden cases are evaluated only after a complete candidate solution is found.

## Compositional transfer

Held-out task:

```python
def solve(xs):
    acc = 0
    for x in xs:
        if x > 0:
            acc += x
    return acc
```

The target combines `agg:sum` learned from `sum_list` and `filter:positive` learned from `count_positive`.

Deterministic five-seed benchmark:

- 600 source-pretraining episodes.
- 250 target-adaptation episodes.
- Seeds 0–4.
- Zero-shot is one deterministic greedy rollout on a copy of the agent, so it cannot perturb later adaptation.

| Metric | Pretrained | Scratch |
|---|---:|---:|
| Seeds finding hidden-generalizing solution during adaptation | 5/5 | 4/5 |
| Mean first-generalizing episode | **49.0** | 142.0 |
| Best first-generalizing episode | **1** | 87 |
| Generalizing solutions during adaptation | 335 | 226 |
| Paired first-solution wins | **5** | 0 |
| Deterministic zero-shot successes | **1/5** | 0/5 |

The zero-shot result is preliminary: one of five pretrained seeds synthesized the correct hidden-generalizing program without any target-task weight update. It is evidence that the shared memory path can compose previously learned factors, not yet evidence of robust zero-shot program synthesis.

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

1. Instrument exact visible-test/oracle calls for compute-normalized Fly vs MCTS comparisons.
2. Add several independent held-out compositions and require zero-shot transfer across them.
3. Learn motor options instead of predefining the `WHEN` family.
4. Add a GRU baseline with matched action/state access.
5. Expand to multiple variables, filters/maps, nested loops and multiple functions.
6. Only after controlled baselines, test FlyWire-derived connectivity motifs against matched synthetic sparse networks.


## MCTS baseline

Mosca now includes a UCT/MCTS baseline over exactly the same `MotorMaze`. MCTS receives the same legal actions and visible-test probe. Hidden cases never affect selection or backpropagation; the search executes its full fixed budget and hidden results are telemetry only.

Five-seed benchmark, 500-simulation budget:

| Task | MCTS success | Mean first hidden-generalizing candidate | Best | Worst |
|---|---:|---:|---:|---:|
| `sum_list` | 5/5 | 44.8 | 9 | 92 |
| `count_positive` | 5/5 | 52.6 | 6 | 108 |
| `max_list` | 5/5 | 120.4 | 33 | 192 |
| `sum_positive` | 5/5 | 65.2 | 31 | 124 |

This is a strong baseline. The pretrained Mosca agent reaches its first hidden-generalizing `sum_positive` solution at 49.0 target-adaptation episodes on average, versus 65.2 MCTS simulations, but Mosca first spent 600 source-pretraining episodes. Episode count and MCTS simulation count are not yet equivalent compute measures because MCTS replays prefixes and both systems invoke the visible-test probe internally.

The next comparison therefore measures visible-test/oracle calls and executed program cases directly rather than treating an episode and a simulation as equal units.
