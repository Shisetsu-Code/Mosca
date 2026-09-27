# Mosca

Experimental program synthesis inspired by navigation and reward learning in *Drosophila*.

Mosca treats programming as navigation through a constrained state space. The learner does not emit arbitrary Python text: CPython owns syntax/AST legality, while the learner selects legal structural or semantic actions.

## Runtime

The world is pinned to **CPython 3.12.14**. The generic AST layer validates the actual runtime AST fields with \`mosca rules\`; Python documentation is not used as training data.

## Architecture

\`\`\`text
problem + tests
      ↓
generic typed AST world
      ↓
semantic motor layer
      ↓
sparse expansion + recurrent history
      ↓
action selection
      ↓
CPython compile/execute
      ↓
test reward + eligibility traces
      └───────────────────────────────┘
\`\`\`

There are three deliberately separate levels:

1. \`PythonMaze\`: original hand-shaped baseline.
2. \`ASTMaze\`: generic low-level grammar with typed holes.
3. \`MotorMaze\`: hierarchical list-reduction curriculum using generated statement-sized motor primitives.

The hierarchy matters. Low-level AST trajectories require roughly 18–26 decisions for the first tasks. The motor layer expresses the same solutions in 5 semantic decisions.

## Motor primitives

The v0 motor curriculum has one accumulator \`acc\`, input list \`xs\`, loop item \`x\`, comparisons against \`0\` or \`acc\`, and generated conditional updates:

\`\`\`text
SET:acc=...
FOR:x:xs
AUG:acc+=...
WHEN:<condition>:SETX
WHEN:<condition>:INC1
WHEN:<condition>:ADDX
END
RETURN:acc
\`\`\`

\`WHEN\` actions are generated compositionally from legal conditions and updates; there is no task-specific \`MAX\`, \`COUNT_POSITIVE\`, or \`SUM\` action.

A provisional program is evaluated during navigation. When it already passes all training tests, goal inhibition freezes semantic changes and permits only block closing / return. This prevents a discovered solution from being destroyed by continued exploration.

## Current benchmark

Deterministic smoke benchmark: 300 episodes, seed 42, CPython 3.12.14.

| Task | Random first solution | Fly first solution |
|---|---:|---:|
| \`sum_list\` | 64 | 147 |
| \`count_positive\` | 253 | 53 |
| \`max_list\` | none in 300 | 93 |

This is an engineering smoke test, not a statistically robust scientific result. Training reward uses only the visible training cases. A separate hidden case set is never used by the reward/probe; the shortest learned solutions for all three tasks score 100% on that hidden set. Broader seed sweeps and held-out tasks are the next validation step.

## Run

\`\`\`bash
python -m pip install -e .[dev]
pytest -q

mosca runtime
mosca rules
mosca ast-reference all
mosca motor-reference all
mosca motor-benchmark --episodes 300 --seed 42 --require-fly-solved --require-fly-generalized
\`\`\`

## Next milestones

1. Run multi-seed statistical comparisons against random search, BFS/MCTS and GRU.
2. Learn reusable motor options instead of predefining the \`WHEN\` option family.
3. Expand beyond reductions: filters, maps, nested loops, multiple variables and functions.
4. Replace task-name input with structured I/O/task sensory features for transfer to unseen tasks.
5. Only then test connectivity motifs derived from FlyWire against matched synthetic sparse networks.
