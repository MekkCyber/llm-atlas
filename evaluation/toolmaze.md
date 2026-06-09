# ToolMaze — benchmark for dynamic replanning under tool failures
*Depth — evaluate tool-using agents on broken tools and non-trivial DAG topologies, not the happy path.*

**TL;DR:** Existing tool-use benchmarks assume tools always succeed. ToolMaze stresses LLM agents with a 2×2 perturbation taxonomy (explicit/implicit × transient/permanent) on top of DAG-shaped task topologies. The headline metric, Perturbation Recovery Rate (PRR), isolates *replanning* skill from raw tool-use accuracy. Implicit semantic failures (a tool returns a wrong-but-plausible answer) drop PRR by ~37%; fault-tolerance scales 3.66× slower with model size than base accuracy does.

**Prereqs:** [rlvr](../post-training/rlvr.md)
**Related:** [ifeval](ifeval.md), [livecodebench](livecodebench.md), [subtlememory](subtlememory.md)

---

## What it is

A benchmark for tool-integrated reasoning (TIR) agents that decouples two axes researchers usually conflate:

- **Topology.** Tasks are DAGs over tool calls — depth and branching are independently tunable. Forces multi-step planning that simple chains miss.
- **Perturbation.** Each tool call may be perturbed along two binary axes:
  - *Explicit vs. implicit*: explicit = tool returns an error / no result; implicit = tool returns a wrong-but-plausible result.
  - *Transient vs. permanent*: transient = retry succeeds; permanent = no retry will help.

The combination gives a 2×2 perturbation matrix on top of a topology grid.

## How it works

- **PRR (Perturbation Recovery Rate).** For each `(topology, perturbation)` cell, PRR = task-success-rate under perturbation / task-success-rate on the clean DAG. PRR is 1 if the agent fully compensates and 0 if it fails entirely.
- **Diagnostic splits.** Per-cell PRR isolates which perturbation type each agent struggles with, separating retry/replanning weaknesses from tool-misuse weaknesses.
- **Scaling test.** Reports PRR-vs-model-size slope alongside accuracy-vs-model-size slope to quantify how slowly robustness improves with scale.

## Why it matters

- Calls out a specific blind spot in TIR evals: "happy-path benchmarks reward smart prompts, not robust agents".
- The implicit-semantic-failure result (over-trust in corrupted outputs) is actionable for RL post-training — train on injected noisy tool outputs to close PRR gaps.
- The 3.66× scaling-gap finding directly contradicts the "wait for bigger models" position on agent robustness.

## Gotchas & tricks

- **DAG generation balance.** Topology distribution matters; an evaluator that loads up on chains misses the planning-heavy regime.
- **Perturbation realism.** Implicit perturbations must be *plausible* (a wrong-looking output isn't a fair stress test); curation effort is non-trivial.
- **Permanent-perturbation tasks need a "give up" exit.** Otherwise agents are penalised for the unsolvable case as if it were a recoverable one.

## Sources

- Paper: *When Tools Fail: Benchmarking Dynamic Replanning and Anomaly Recovery in LLM Agents* — Zhu, Ma, Shen, Li, Zhao, Wang, Yan, Yin — 2026 — [arXiv:2606.05806](https://arxiv.org/abs/2606.05806)
- Code: https://github.com/Zhudongsheng75/ToolMaze
