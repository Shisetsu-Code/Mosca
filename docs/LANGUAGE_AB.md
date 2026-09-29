# Language A/B Protocol

Mosca should compare programming languages without changing the learner.

A language backend implements the `ProgramWorld` contract:

- `observe()`
- `valid_actions()`
- `step(action)`
- `evaluate_hidden()`
- `source()`

The task definitions, visible/hidden I/O cases, random seeds, learner topology,
training budget and hardware must remain identical between language runs.

## Primary metrics

Training:

- episodes to first visible solution
- episodes to first hidden-generalizing solution
- visible oracle calls
- visible case executions
- CPU/GPU time
- peak memory
- state transitions
- action branching factor

Final program:

- hidden-test generalization
- semantic action count
- source/IR node count
- generated program runtime
- generated program memory use
- compiled artifact size when meaningful

Transfer:

- zero-shot success
- adaptation episodes
- adaptation oracle calls
- retained performance on previously learned concepts

## Fairness rules

Do not give one language task-specific macros that the other language does not
have. Language-specific syntax may differ, but the semantic capability exposed
at a given curriculum level should be matched.

Do not count parser punctuation, indentation, or spelling as neural decisions
when those are deterministic language mechanics. Measure semantic decisions.

Use two comparisons:

1. **Fixed interaction budget:** same number of training episodes/oracle calls.
2. **Fixed wall/compute budget:** same CPU/GPU budget.

For the future custom language, register its world factory under a new name.
The Mosca learner should not need modification.
