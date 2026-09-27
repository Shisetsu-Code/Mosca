# Mosca

Experimental program synthesis inspired by navigation and reward learning in *Drosophila*.

Mosca treats programming as navigation through a constrained state space. The learner never writes arbitrary Python text: CPython owns syntax/AST legality, while the learner selects legal structural or semantic actions.

## Runtime

The world is pinned to **CPython 3.12.14**. Python documentation is not used as model training data.

## Architecture

```text
visible I/O examples
      ↓
sensory signature (no task name)
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

The current learner no longer receives `sum_list`, `max_list`, etc. as neural input. It receives a deterministic sensory sketch derived only from visible input/output examples: lengths, signs, low-order numeric buckets, and relations such as equality with aggregate statistics.

Motor actions are also factorized. For example:

```text
WHEN:x>0:ADDX
```

is represented through reusable components such as `WHEN`, `x`, `>`, `0`, and `ADDX`, in addition to the exact action. This allows experience from different programs to share weights.

## Environments

1. `PythonMaze`: original hand-shaped baseline.
2. `ASTMaze`: generic low-level grammar with typed holes.
3. `MotorMaze`: hierarchical list-reduction curriculum using generated statement-sized motor primitives.

The hierarchy matters. Low-level AST trajectories require roughly 18–26 decisions for the first tasks. The motor layer expresses the same solutions in about 5 semantic decisions.

## Hidden evaluation

Training reward uses only visible cases. Hidden cases are evaluated only after a complete training solution is found.

## Transfer experiment

The first held-out compositional task is `sum_positive`:

```python
acc = 0
for x in xs:
    if x > 0:
        acc += x
return acc
```

It recombines behavior needed by the trained `sum_list` and `count_positive` tasks. The benchmark compares adaptation after pretraining on core tasks against the same agent trained from scratch on `sum_positive`.

## Run

```bash
python -m pip install -e .[dev]
pytest -q

mosca runtime
mosca rules
mosca motor-benchmark --episodes 300 --seed 42 --require-fly-solved --require-fly-generalized
mosca motor-multiseed --episodes 150 --seeds 0,1,2
mosca transfer-benchmark --pretrain-episodes 900 --adapt-episodes 300 --seed 42
mosca transfer-multiseed --pretrain-episodes 600 --adapt-episodes 250 --seeds 0,1,2,3,4
```

## Next milestones

1. Quantify transfer over many seeds and multiple unseen compositions.
2. Add MCTS and GRU baselines over exactly the same motor action space.
3. Learn reusable motor options instead of predefining the `WHEN` option family.
4. Expand beyond reductions: filters, maps, nested loops, multiple variables and functions.
5. Only after controlled baselines, test FlyWire-derived connectivity motifs against matched synthetic sparse networks.


## Two-timescale contextual memory

Successful trajectories can partially consolidate eligibility-trace synapses into slow weights. Slow memory is keyed by the sensory signature that created it, so source-task memories do not directly score actions on an unseen task.

The slow memory influences action selection, but the TD prediction error is computed only from fast shared weights. This separation was important: global slow memory caused negative transfer, and contextual slow memory coupled into TD still slowed discovery.

Five-seed transfer benchmark (`sum_positive`, 600 source-pretraining episodes, 250 adaptation episodes):

| Metric | Transfer | Scratch |
|---|---:|---:|
| Seeds finding hidden-generalizing solution | 5/5 | 4/5 |
| Mean first-generalizing episode | 68.2 | 142.0 |
| Total generalizing solutions | 341 | 188 |
| Paired first-solution wins | 5 | 0 |

The prior version without slow contextual memory reached 73.4 mean first-generalizing episodes and 256 total generalizing solutions under the same five-seed transfer setup. These are small synthetic tasks, so the result is evidence for the architecture direction, not a claim of general program synthesis.
