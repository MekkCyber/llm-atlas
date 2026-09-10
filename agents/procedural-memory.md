# Procedural memory (Procedural Graphs)
*Depth — external memory of "what to do" as a self-evolving graph over procedures, used to bias agent action selection at each step.*

**TL;DR:** Knowledge graphs organise factual knowledge as (entity, relation, entity); Procedural Graphs organise procedural knowledge as (procedure, relation, procedure). At each decision step the agent's active node is localised, a guidance model turns the surrounding subgraph into step-level advice, and an LLM refiner edits the graph after each rollout — keeping edits that hold up on held-out validation, retaining rejected edits as anti-patterns.

**Prereqs:** *(agents folder has only a README today; no depth prereqs yet)*
**Related:** [feedback-enriched-environments.md](feedback-enriched-environments.md)

---

## What it is

An external memory dedicated to procedural knowledge — the "what to do, in what order, and under which conditions" that long-horizon agents otherwise re-derive from raw trajectories. Distinct from embedding-store memory (nearest-neighbour text recall) and from parametric memory (fine-tuning). The graph is mutable and grows with experience.

## How it works

**Data model.** A directed graph whose nodes are procedures (typed with description, preconditions, expected effect) and whose edges are typed relations: `sequence`, `alternative`, `precondition`, `subroutine`, `guarded_by`, `anti_pattern`.

**Inference-time loop.** At each decision step for a task:

1. **Localise.** Identify the agent's active node from the current state and history (embedding match + graph consistency check).
2. **Guide.** A guidance model reads a bounded subgraph around the active node and emits step-level advice — bias only, not a hard action.
3. **Act.** The solver LLM chooses the next action conditioned on the advice; unconstrained choice is retained but soft-biased.

**Learning-time loop** (self-evolution). After each rollout:

1. **Contrast.** An LLM refiner compares failed against successful trajectories that touched overlapping subgraphs.
2. **Propose edits.** Add / remove / retype nodes and edges; edit attributes (preconditions, expected effect).
3. **Validate.** Apply the edit only if held-out validation performance is preserved or improved; retain **rejected** edits as `anti_pattern` edges so the guidance model actively discourages rediscovery.

Starts from a minimal skeleton (or a hand-designed expert prior); grows to match or surpass hand-designed graphs.

## Why it matters

- **Auditable memory.** Unlike opaque parametric memory, procedural edges are inspectable and editable.
- **Composable with any solver.** Guidance is a soft bias; no changes to the underlying agent LLM.
- **Robust to bad priors.** The refiner can repair a flawed expert graph, not just extend a good one.
- **Anti-patterns matter.** Keeping rejected edits prevents the loop from re-proposing them and stabilises long training runs.

## Gotchas & tricks

- **Localisation is the hard part.** Bad active-node detection propagates bad guidance; combine embedding match with graph-topology plausibility.
- **Subgraph size for guidance.** Too small → thin advice; too large → the guidance model over-fits to distant nodes. The paper uses bounded k-hop neighbourhoods.
- **Validation must be held-out.** Accepting edits by training-set performance leads to graph bloat and eventual regression.
- **Doesn't replace RL.** Procedural graphs help with *what to try*; they don't provide the reward signal that RL post-training does — the two compose.

## Sources

- Lu, Chen, Wu, Arık — *Procedural Graphs: Self-Evolving Execution Structures for LLM Agents* — 2026 — https://arxiv.org/abs/2609.09153
