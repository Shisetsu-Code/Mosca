# Mosca

Experimental program synthesis inspired by navigation and reward learning in *Drosophila*.

Mosca treats programming as navigation through a constrained state space rather than next-token prediction. The learner never emits arbitrary Python text: CPython owns syntax and AST legality, while the learner selects legal structural or semantic actions.

## Runtime and program worlds

The Python reference world is pinned to **CPython 3.12.14**. Python documentation is not used as model training data.

v0.12 adds a second, independent `mosca` world. It executes the same semantic motor programs directly over a tiny native IR: no Python AST construction and no `compile()`. Both worlds implement the same `ProgramWorld` contract, so the learner, MCTS, transfer suite and resource profiler can run unchanged.

```bash
mosca worlds
# python
# mosca
```

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

## First language-world A/B

The first independent backend is now implemented. The initial comparison deliberately keeps the **same action vocabulary and branching factor** in both worlds. This isolates execution/runtime effects before changing the language itself.

Five-seed resource benchmark, identical trajectories and oracle counts:

| Metric | CPython world | Native Mosca world | Change |
|---|---:|---:|---:|
| Full pretrain + transfer CPU | 14.73 s | **13.40 s** | **-9.1%** |
| MCTS CPU | 0.406 s | **0.304 s** | **-25.0%** |
| Random-search CPU | 0.470 s | **0.313 s** | **-33.5%** |
| Transfer episodes by seed | 1, 91, 42, 24, 84 | 1, 91, 42, 24, 84 | identical |
| Transfer oracle calls by seed | 5, 526, 257, 124, 444 | 5, 526, 257, 124, 444 | identical |

Individual runner timings vary, so the important result is the paired semantic equivalence: every seed followed the same learning/search trajectory while the backend changed.

This **does not yet measure the advantage of a better language grammar**. The next A/B changes the native world's action structure/branching while holding the task semantics and learner fixed.

For Python versus a compact custom language, hold constant:

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

The custom language plugs into the same world contract so only the programming environment changes.

## Native Mosca syntax

A discovered reduction can now be rendered independently of Python:

```text
solve xs
  acc := 0
  each x in xs
    when x>0 => acc += x
  end
  yield acc
end
```

The current native world still exposes the canonical motor actions used by Python. This is intentional: v0.12 establishes backend equivalence first.

## Next milestones

1. Add a **compact native action grammar** and measure branching-factor/decision-depth tradeoffs against Python.
2. Expand beyond one-accumulator list reductions.
3. Learn motor options instead of predefining the `WHEN` family.
4. Add a GRU baseline with matched state/action access.
5. Revisit adaptive factor gates after a larger concept curriculum.
6. Compare synthetic sparse topology against FlyWire-derived motifs after the benchmark suite is broader.
