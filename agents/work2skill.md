# Work2Skill
*Depth — reusable agent skill induction from aligned human GUI demonstrations.*

**TL;DR:** Human demos encode procedural knowledge, but using them as agent *skills* requires separating workflow rules (always open the file menu first) from task-specific details (filename X, folder Y). Work2Skill pairs **WorkRecorder** (synchronized screen + mouse/keyboard + accessibility-tree captures) with an *agentic* skill-induction process that inspects evidence on demand to produce skills with explicit workflow rules, step dependencies, and completion checks. The induced skills raise agent success by 3–11 points over no-skill baselines on MyPCBench and OSWorld.

**Prereqs:** [rl-prompt-curation](../post-training/rl-prompt-curation.md)
**Related:** (none in graph yet; cross-ref SoftwareGym once this page lands)

---

## What it is

A system with two halves:

1. **WorkRecorder** — a capture tool that aligns screen video, keyboard/mouse events, and the OS accessibility tree. The alignment matters: pure screen recording loses structured context (which widget was clicked); pure event logs lose visual appearance.
2. **WorkDemo-350** — a dataset of 350 verified human-performed desktop demonstrations collected with WorkRecorder, screened for completeness and task success.
3. **Agentic skill induction** — a procedure that reads demos and produces reusable skills, where each skill is a structured description of a workflow with its rules, dependencies, and success checks.

## How it works

The inducer is itself an agent, not a one-shot summarizer:

- It reads a demo end-to-end to identify the workflow boundary.
- It inspects evidence on demand: replays specific steps, re-queries the accessibility tree, cross-references demos.
- It produces a skill *abstraction* — step dependencies in a DAG, parametric fields (filename, folder, …) separated from fixed workflow rules, and a completion check that the executor can later verify.

Executing agents then use these skills as retrievable procedures rather than raw demos, which is what gives the 3–11-point lift over no-skill baselines across four different execution models.

## Why it matters

Complementary to environment-synthesis approaches like SoftwareGym: SoftwareGym scales the *environment* count; Work2Skill scales the *expertise per environment*. Together they outline a hybrid training recipe where synthetic apps provide breadth and human demos provide the harder, deeper knowledge of real workflows.

## Gotchas & tricks

- **Alignment is harder than it looks.** OS accessibility trees have latency and go out of sync with the visual state; the recorder must handle both.
- **Skill granularity.** Too fine → not reusable; too coarse → brittle. The abstraction step is where most quality lives.
- **Four execution models tested.** The lift varies by model; stronger executors need less skill scaffolding, so the technique shines most for mid-tier agents.

## Sources

- Paper: *Work2Skill: Capturing Human GUI Experience for Reusable Agent Skill Induction* — Guo, Zhang, Li et al., 2026 — [arXiv:2610.12984](https://arxiv.org/abs/2610.12984)
- Code: [github.com/china-qijizhifeng/Work2Skill](https://github.com/china-qijizhifeng/Work2Skill)
