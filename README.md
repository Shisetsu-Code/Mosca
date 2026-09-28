# Mosca

Experimental program synthesis inspired by navigation and reward learning in *Drosophila*.

Mosca treats programming as navigation through a constrained state space rather than next-token prediction. The learner never emits arbitrary Python text: CPython owns syntax and AST legality, while the learner selects legal structural or semantic actions.

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
MotorMaze / typed AST
      ↓
CPython compile + execute
      ↓
visible-test reward
```

The implementation uses NumPy dense fast-weight / eligibility vectors, while contextual and conceptual long-term memories remain sparse.

## Environments

1. `PythonMaze`: original hand-shaped baseline.
2. `ASTMaze`: generic typed-hole grammar.
3. `MotorMaze`: hierarchical semantic action space.

The first benchmark programs require about five motor decisions versus roughly 18–26 low-level AST decisions.

## Hidden-test policy

Training reward and probe use only visible cases. Hidden cases never influence action selection, TD updates, MCTS backpropagation, or stopping decisions in the fixed-budget search benchmarks. They are used only to evaluate completed visible solutions.

## Concept and factor memory

Stable relations inferred from visible input/output examples produce broad concepts such as:

```text
agg:sum
agg:count
agg:max
filter:all
filter:positive
filter:negative
role:identity_zero
role:list_reduce
```

v0.10 keeps the v0.9 broad concept memory and adds an independent factor channel. Factor memory learns reusable motor pieces while gated off during source pretraining. On a novel target, the validated defaults are `role_factor_mix=0.30`, `agg_factor_mix=0.15`, and `filter_factor_mix=0.00`. The model does not receive task names such as `sum_list` or `count_positive` as neural input.

## Multi-target transfer suite

Source tasks remain:

- `sum_list`
- `count_positive`
- `max_list`

Held-out compositions:

| Target | Recombined concepts |
|---|---|
| `sum_positive` | `agg:sum` + `filter:positive` |
| `count_all` | `agg:count` + `filter:all` |
| `max_positive_or_zero` | `agg:max` + `filter:positive` |

Five deterministic seeds, 600 source-pretraining episodes and 250 target-adaptation episodes:

| Target | Transfer success | Scratch success | Mean first solution* | Scratch mean* | Transfer visible-oracle calls* | Scratch calls* | Zero-shot |
|---|---:|---:|---:|---:|---:|---:|---:|
| `sum_positive` | 5/5 | 4/5 | **48.4** | 142.0 | **271.2** | 622.75 | **1/5** |
| `count_all` | **3/5** | 1/5 | 17.7 | 77.0 | **113.3** | 253.0 | 0/5 |
| `max_positive_or_zero` | 5/5 | 5/5 | **43.0** | 78.0 | **245.8** | 299.0 | 0/5 |

* Means are over seeds that found a hidden-generalizing solution, so success rate must be read with the mean.

Compared with v0.9, the validated factor channel keeps `sum_positive` roughly neutral on first-solution latency, raises `count_all` transfer robustness from 2/5 to 3/5 seeds, and reduces `max_positive_or_zero` mean first solution from 72.4 to 43.0 episodes. Robust zero-shot composition is still not solved.

## MCTS baseline

Mosca includes UCT/MCTS over exactly the same `MotorMaze`.

Five-seed, 500-simulation baseline:

| Task | MCTS success | Mean first hidden-generalizing candidate |
|---|---:|---:|
| `sum_list` | 5/5 | 44.8 |
| `count_positive` | 5/5 | 52.6 |
| `max_list` | 5/5 | 120.4 |
| `sum_positive` | 5/5 | 65.2 |

For `sum_positive`, MCTS reaches the first hidden-generalizing candidate after about **448.6 visible-oracle calls** on average. The pretrained Mosca agent needs about **278.0 target-task oracle calls**, but Mosca first pays a separate source-pretraining cost. The repository therefore reports both marginal target cost and pretraining cost instead of equating one MCTS simulation with one learning episode.

## Performance

The original Python implementation spent most of its CPU time repeatedly scoring sparse action components. The current hot path uses:

- dense NumPy fast weights;
- dense vectorized eligibility traces;
- batched shared-component Q evaluation;
- cached sparse encoder token hashes;
- probe evaluation without AST `deepcopy`.

On the controlled cProfile workload (`150` pretraining + `80` adaptation episodes), cumulative runtime fell from about **22.34 s to 2.29 s** on the same GitHub Actions runner class.

In the five-seed resource benchmark, target adaptation preserves exactly the same episode/oracle counts while substantially reducing CPU cost. Traced peak memory remains in the single-digit MiB range at the current 8,192-unit expansion width.

## Run

```bash
python -m pip install -e .[dev]
pytest -q

mosca runtime
mosca rules

mosca motor-benchmark \
  --episodes 300 \
  --seed 42 \
  --require-fly-solved \
  --require-fly-generalized

mosca transfer-suite \
  --pretrain-episodes 600 \
  --adapt-episodes 250 \
  --seeds 0,1,2,3,4

mosca mcts-multiseed \
  --simulations 500 \
  --seeds 0,1,2,3,4

mosca resource-benchmark \
  --seed 0 \
  --pretrain-episodes 600 \
  --adapt-episodes 250 \
  --mcts-simulations 500
```

## Current conclusions

- Hierarchical motor actions are dramatically easier to search than raw AST-node actions.
- Sparse recurrent learning reliably transfers useful structure to unseen compositions.
- Transfer can reduce target-task oracle calls relative to scratch and, on `sum_positive`, relative to MCTS.
- Pretraining is not free; MCTS remains cheaper for a single isolated problem.
- Simply increasing concept-memory influence does not make zero-shot robust.
- Separating broad memory from a low-weight factor channel reduces interference.
- Structural-role and aggregation factors help when enabled only on novel targets; filter factors remain disabled by default after the ablation.

## Next milestones

1. Learn the factor gates instead of fixing them by ablation.
2. Add more independent source/target concept combinations.
3. Learn motor options instead of predefining the `WHEN` family.
4. Add a GRU baseline with matched state/action access.
5. Expand to multiple variables, filters/maps, nested loops and multiple functions.
6. Compare synthetic sparse topology against FlyWire-derived motifs only after the controlled baselines are strong.
