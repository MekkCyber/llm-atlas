# Workspace Synthesis for Working-Agent SFT
*Depth — build training data for working agents by assembling real-file workspaces and projecting tasks + verifiers from an evidence graph over them.*

**TL;DR:** Training "working agents" (OpenHands-style coding/office agents) needs tasks over real files **with executable verifiers**. Prior pipelines either generate files with models (unrealistic) or build tasks on real files without verifiers (unchecked). **GraphForge** (Su et al., 2026) seeds from occupation-grounded descriptions, assembles a real-file workspace per seed, constructs an evidence graph over file relations, and derives both the task statement and the verifier rubric from that graph — so every rubric item is backed by concrete files.

**Prereqs:** [README.md](./README.md), [../post-training/_post-training.md](../post-training/_post-training.md)
**Related:** [../post-training/rejection-sampling.md](../post-training/rejection-sampling.md) · [../data/quality-filtering.md](../data/quality-filtering.md)

---

## What it is

A pipeline for synthesizing (task, workspace, rubric, trajectory) tuples where:

- **Workspace** = a real-file directory (spreadsheets, docs, code), chosen for the task's domain.
- **Evidence graph** = nodes are files, edges are relations the task needs (summarizes, references, updates, feeds-into).
- **Task statement** = natural-language goal projected from a sub-graph.
- **Rubric** = per-criterion verifier, each clause anchored to the specific files that must change / be read.
- **Trajectory** = an agent's rollout against the workspace, scored by the rubric.

Rubrics act both as training-time quality filters (keep only passing trajectories) and as a reward signal for self-improvement rounds.

## How it works

1. **Occupation-grounded seeds.** A seed describes a job function ("quarterly budgeting at a mid-size retailer") that steers which file types and relations populate the workspace.
2. **Workspace assembly.** Pull real files matching the seed (anonymized spreadsheets, policy docs, code repos). Avoid model-generated contents.
3. **Evidence graph construction.** For each pair of files, infer the relation the task needs (data source, dependency, summary target). Store as a labeled DAG.
4. **Task / rubric projection.** The natural-language task is generated from one sub-graph; the rubric is a conjunction of per-edge checks projected from the same sub-graph. Both reference the files by name.
5. **Executability validation.** A rollout agent attempts the task once; a revision agent repairs mismatches between the task text and the files before trajectories are collected at scale.
6. **Trajectory collection.** Standard rollout against the workspace; rubric gives a per-criterion pass/fail. Fine-tune (or RFT) on passing trajectories.

## Why it matters

- **Verifiers for working agents become feasible.** This was the main bottleneck; evidence graphs give executable rubrics for free alongside task text.
- **Realistic workspaces.** Real files avoid the sterility of synthetic workspaces and the embedded-tells LLMs leave in generated spreadsheets.
- **Doubles as a reward signal.** The same rubric scores the policy's own rollouts during rejection fine-tuning — self-improvement without a learned RM.

Reported results (Qwen3.6-27B fine-tuned on 2,169 trajectories):

- GDPVal under OpenHands: 1380.0 → 1445.7 (+65.7).
- Workspace-Bench-Lite: 56.0 → 63.7 (+7.7).
- SpreadsheetBench II under Claude Code: 10.3 → 24.0 (+13.7).

Rejection fine-tuning on the SFT model's own rollouts, rubric-filtered, yields additional stacked gains.

## Gotchas & tricks

- **Seed diversity controls coverage.** Narrow seeds produce narrow workspaces; expect the pipeline to under-serve rare occupation types unless seeds explicitly cover them.
- **File anonymization.** Real workspaces often leak PII; the pipeline needs a redaction pass before public release.
- **Rubric brittleness.** Projecting rubrics from graphs can produce clauses that trip on cosmetic differences (whitespace, column order); add tolerant matchers.
- **Evidence-graph construction cost.** Pairwise relation inference scales $O(n^2)$ in workspace size; practitioners cap workspaces at ~20 files or use heuristic edge filters.
- **Rubric-as-reward overfits** to format if the rubric is too shallow. The paper's revision pass is the main defense; without it, agents learn to game the rubric rather than solve the task.

## Sources

- Paper: *GraphForge: Training Working Agents with Graph-Anchored Workspace Synthesis* — Su et al., 2026 — [arXiv:2609.38923](https://arxiv.org/abs/2609.38923).
- Related: *SWE-Gym*, *OpenHands* workspace schemas (reference for the agent runtime).
- Related: *Workspace-Bench*, *SpreadsheetBench II*, *GDPVal* (evaluation benches the paper reports on).
