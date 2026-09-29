# Mosca

Experimental program synthesis inspired by navigation and reward learning in *Drosophila*.

Mosca treats programming as navigation through a constrained state space rather than next-token prediction. The learner never emits arbitrary Python text: CPython owns syntax and AST legality, while the learner selects legal structural or semantic actions.

## Runtime

The reference world is pinned to **CPython 3.12.14**. Python documentation is not used as model training data.

## Architecture

```text
visible I/O examples
      ↓
fast sensory signature
      ↓
sparse recurrent policy
      │
      ├── fast TD(lambda) weights
      ├── exact-context slow memory
      ├── broad concept memory
      └── gated factor memory
             ↓
       semantic MotorMaze
             ↓
       typed AST / CPython
             ↓
       compile + execute
             ↓
       visible-test reward
```

The model does not receive task names as neural input.

## Source curricula

The stable default remains the original **core** curriculum:

- `sum_list`
- `count_positive`
- `max_list`

v0.11 also includes an optional **expanded** curriculum:

- the three core tasks
- `count_negative`
- `min_list`

Use `--source-curriculum expanded` when testing the larger concept bank.

## Held-out compositions

The transfer suite now contains five held-out tasks:

- `sum_positive`
- `sum_negative`
- `count_all`
- `max_positive_or_zero`
- `min_negative_or_zero`

Visible cases drive training. Hidden cases are evaluation-only and never enter reward, TD updates, MCTS backpropagation or stopping decisions.

## Factor memory

The current validated target-time defaults are:

```text
role_factor_mix   = 0.30
agg_factor_mix    = 0.15
filter_factor_mix = 0.25
```

The filter gate is intentionally asymmetric in v0.11:

- `filter:negative` is **factor-only**;
- `filter:positive` and `filter:all` retain the established broad-memory path.

This matters because `count_negative` teaches both “x < 0” and “add 1”. Broad transfer can incorrectly drag the update operation into `sum_negative`. The factor-only path transfers the condition without the source task's update.

The `min_negative_or_zero` target is decomposed as `agg:min + role:identity_zero`, not as an explicit negative filter, because its correct comparison is `x < acc`.

### Expanded-curriculum result

With 1,000 source-pretraining episodes, 250 adaptation episodes and five seeds:

- baseline expanded curriculum: `sum_negative` generalized in **4/5** seeds;
- negative-filter factor-only at 0.25: **5/5** seeds;
- mean first generalizing solution improved from **81.2 → 73.4 episodes**;
- mean visible-oracle calls improved from **455.5 → 409.4**;
- the other four targets were unchanged seed-for-seed in the final controlled check.

This is the strongest v0.11 change: a localized transfer gain without observed regressions in the rest of the suite.

## Performance

The hot path currently uses:

- dense NumPy fast weights;
- dense vectorized eligibility traces;
- batched shared-component Q evaluation;
- cached sparse encoder hashes;
- AST probe evaluation without `deepcopy`.

On the controlled cProfile workload, cumulative runtime was reduced from roughly **22.34 s to 2.29 s** on the same GitHub Actions runner class.

## MCTS baseline

Mosca also includes UCT/MCTS over the same `MotorMaze`. This remains the main classical-search baseline. Mosca reports source-pretraining cost separately from marginal target-task adaptation cost.

## Run

```bash
python -m pip install -e .[dev]
pytest -q

# Stable/core behavior
mosca transfer-suite \
  --pretrain-episodes 600 \
  --adapt-episodes 250 \
  --seeds 0,1,2,3,4

# Expanded concept curriculum
mosca transfer-suite \
  --pretrain-episodes 1000 \
  --adapt-episodes 250 \
  --seeds 0,1,2,3,4 \
  --source-curriculum expanded
```

## Language A/B direction

The next major comparison is language-level rather than only architecture-level.

For Python versus a custom language, hold constant:

- Mosca topology and hyperparameters;
- semantic tasks and visible/hidden I/O cases;
- seeds;
- interaction/oracle budgets;
- hardware/compute budget.

Measure:

- episodes and oracle calls to first generalizing solution;
- CPU/GPU time and memory;
- action branching factor;
- semantic decisions per solution;
- final program runtime and size;
- zero-shot and adaptation transfer.

The custom language should plug into the same world contract so only the programming environment changes.

## Next milestones

1. Add a pluggable language-world backend and run Python versus the new language.
2. Expand beyond one-accumulator list reductions.
3. Learn motor options instead of predefining the `WHEN` family.
4. Add a GRU baseline with matched state/action access.
5. Revisit adaptive factor gates after a larger concept curriculum.
6. Compare synthetic sparse topology against FlyWire-derived motifs after the benchmark suite is broader.
