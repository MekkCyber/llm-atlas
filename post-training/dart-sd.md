# DART-SD
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A self-distillation recipe for multi-turn tool-calling agents where sub-goals are **order-independent** and the valid solution space forms a combinatorial "diamond lattice". Instead of imitating full teacher trajectories (which collapses the diamond and penalizes alternative correct paths), DART-SD models rollouts as a graph, finds the exact step where a rollout diverged from a good trajectory, retrieves a recovery reference, and computes loss **only on the generated recovery tokens** while masking gradients on the valid reasoning prefix.

**Prereqs:** [rejection-sampling.md](rejection-sampling.md), [rlvr.md](rlvr.md)
**Related:** [../agents/README.md](../agents/README.md) · [grpo.md](grpo.md) · [../post-training/fine-tuning/README.md](fine-tuning/README.md)

---

## What it is

When a task has many order-independent sub-goals (a directory of files to read, a set of tools that commute), the set of correct completions is not one trajectory but many. Standard full-trajectory imitation over a single teacher rollout scores *all* other valid orderings as wrong, collapsing exploration diversity into whatever the teacher happened to choose. DART-SD fixes the imitation objective to respect this topology.

## How it works

Three ingredients:

1. **Interaction-State Transition Graph (ISTG).** Rollouts (successes and failures) are folded into a converging DAG over abstracted interaction states rather than raw token sequences, so different orderings that reach the same state merge into one node.
2. **Critical Topological Breakpoint (CTB) detection.** Given an autonomous rollout, find the earliest node in the ISTG where the trajectory diverged from any success-supported subgraph. That step, not the whole trajectory, is what needs correction.
3. **Localized self-distillation.** Retrieve a **success-supported recovery reference** from the neighbouring successful subgraph, generate a recovery continuation, and compute the training loss **only on the recovery tokens** — the valid prefix before the CTB is loss-masked so its gradients don't overwrite legitimate alternative reasoning.

The result is a progressive self-distillation loop: sample rollouts → build/update ISTG → localize CTB → retrieve reference → SFT on recovery step only.

## Why it matters

- Beats full-trajectory imitation baselines on complex multi-turn tool-calling benchmarks — the improvement comes from **not destroying** valid alternative solutions.
- Preserves policy diversity across rollouts, which is exactly what downstream RL (RLVR / GRPO) needs to work well.
- Generalizes: any task where sub-goals commute (research agents, retrieval fan-out, batch tool use) has the same topological structure and benefits from the same fix.

## Gotchas & tricks

- Building the ISTG requires a state abstraction; too fine and every trajectory is a unique path (no merging), too coarse and CTB detection is unreliable.
- Loss masking on the prefix is essential — without it you re-introduce the topological collapse the method fights.
- Composes with GRPO downstream: the diversity DART-SD preserves is exactly the group-of-samples GRPO needs to have non-degenerate advantages.

## Sources

- Paper: *DART-SD: Diamond-topology Aware Retrieval and Tuning for Self-Distillation of Multi-Turn Tool-Calling Agents* — Xu et al., 2026 — [arxiv](https://arxiv.org/abs/2608.18524)
