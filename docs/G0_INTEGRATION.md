# Mosca + G0 integration contract

Status: design direction for the post-bootstrap system.

## 1. Purpose

Mosca is not intended to become a general-purpose language model that learns the historical software ecosystem.

Its target environment is G0: a deliberately small, graph-first backend/data platform in which the compiler and runtime expose structure directly.

The working hypothesis is that Mosca may remain comparatively weak in general intelligence while becoming highly efficient inside a constrained environment whose legal states, dependencies, effects and costs are observable.

## 2. G0 is the target world

Python remains useful as a controlled experimental/reference world during early Mosca development.

It is **not** the target compatibility model.

The target G0 ecosystem does not require compatibility with:

- Python
- SQL
- HTTP
- HTML/CSS/JavaScript
- DOM/browser engines
- JSON/ORM/DTO layering
- existing web application conventions

G0 is expected to own its storage, transport, service/runtime model and eventual client environment.

Mosca experiments must therefore avoid optimizing toward accidental complexity that exists only because of legacy ecosystem compatibility.

## 3. Environment-assisted intelligence

Mosca should exploit information that G0 already knows instead of relearning it from source text.

Useful compiler/runtime inputs include:

- typed AST/GIR
- dataflow
- control/iteration structure
- ownership/lifetime constraints
- effects/capabilities
- authorization/storage constraints
- hot nodes and critical paths
- latency
- memory
- I/O
- synchronization boundaries
- benchmark/test outcomes

The environment should reduce the search path before increasing model complexity.

## 4. Structural adaptation, not weights only

Mosca research must not assume that continuously tuning neural weights inside a fixed topology is the best optimization mechanism.

Candidate adaptation mechanisms may include:

- changing connectivity;
- moving or regrouping functional units;
- shortening frequently useful paths;
- pruning ineffective paths;
- adding reusable subcircuits/options;
- specializing regions for recurring graph motifs;
- changing routing/gating;
- retaining fast learned weights where useful.

Topology and routing are therefore experimental variables, subject to the same controlled measurement discipline as weights and hyperparameters.

## 5. Empirical loop

The preferred loop is:

```
G0 graph/problem
    -> constrained legal actions
    -> Mosca candidate
    -> compile
    -> validate
    -> execute
    -> test
    -> profile
    -> attribute gain/loss to graph regions
    -> update topology/policy/search strategy
```

A change is not preferred because the project assumes it should be better.

It must be observed, tested and measured.

## 6. Selective optimization

G0 exposes AST/dataflow/profiling information so Mosca does not need to search the whole program after every observation.

If a small graph region dominates cost, search may be restricted to that region while the rest of the program remains frozen.

Useful metrics include:

- semantic decisions per valid improvement;
- state transitions per valid improvement;
- activations/computation per valid improvement;
- route/path length;
- branching factor;
- oracle/test executions;
- compiler rejections;
- runtime latency;
- memory;
- energy/hardware counters where available;
- regressions outside the targeted graph region.

A particularly important long-term metric is **computational work per useful program modification**, not only benchmark score.

## 7. End-to-end data structures

G0's intended native stack lets the same typed logical structure flow across storage, backend, transport and client semantics.

Mosca should reason about these as one dataflow when possible.

For example:

```
client field/event
    -> typed mutation
    -> policy check
    -> storage transition
    -> dependent computations
    -> synchronized client state
```

The goal is to eliminate artificial translation tasks such as repeatedly recreating the same boolean/field in SQL, ORM, DTO, JSON and frontend state.

## 8. No technical-debt training target

Do not treat familiarity with existing APIs, frameworks or protocol quirks as intelligence that Mosca must acquire unless a G0-native requirement independently needs the same semantic capability.

The long-term objective is a small canonical primitive set and a native ecosystem designed to make legal solutions short, observable and measurable.

## 9. Relationship to current language A/B tests

Current Python/native-world A/B tests remain useful for isolating the effect of action grammar and branching factor.

They should be interpreted as experimental scaffolding, not as a commitment to Python compatibility.

The future comparison of interest is increasingly:

```
legacy/general-purpose environment
vs
G0-native graph environment
```

while holding learner compute and task semantics controlled.

## 10. Continuation rule for future work

When extending Mosca or G0, do not start from the assumption that external compatibility is required.

First ask:

1. What semantic capability is actually needed?
2. Can it be represented once in G0's graph/type/effect system?
3. Can compiler/runtime structure reduce Mosca's search space?
4. Can the change be measured locally and end to end?
5. Does it introduce redundant representations or compatibility debt?

If the answer to (5) is yes, the default is to redesign rather than preserve the debt.
