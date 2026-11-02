# Agent Task Synthesis (Backward-Planning + Probe-Repair)

*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A recipe for synthesizing executable *professional-work* tasks for agent training at scale. Plan *backward* from an intended output through an "information support graph" of required facts and tools; then use **probe executions** of a draft task to detect missing or inconsistent environment context, and auto-repair the environment without changing the task's intent. Introduced in WorkFactory, which built 112,686 grounded tasks across 612 occupations and fine-tuned Qwen (4B/27B/35B-A3B) into the WorkPilot family.

**Prereqs:** [../data/_data-curation](../data/_data-curation.md)
**Related:** [../post-training/rejection-sampling](../post-training/rejection-sampling.md), [../post-training/rl-prompt-curation](../post-training/rl-prompt-curation.md)

---

## What it is

Agent training data is bottlenecked by *grounded, executable* tasks: a prompt alone is cheap, but a prompt plus an environment where a specific trajectory of tool calls produces a specific output is expensive to construct and almost never scales past a few hundred hand-built scenarios. WorkFactory inverts the usual pipeline:

- **Forward** (traditional): write a task, build an environment, hope the task is actually solvable in it.
- **Backward** (WorkFactory): choose a desired output, derive the information graph that would be needed to produce it, probe an environment for that information, and repair gaps.

## How it works

### 1. Backward intent → support graph

Given a professional-intent description (e.g. "generate a Q3 revenue report for an SMB electronics retailer"), the planner produces an **information support graph**: nodes are facts, documents, tool outputs, and intermediate computations; edges are "A supports B" dependencies. The planner is itself an LLM prompted with a schema and example graphs.

### 2. Probe execution

A draft agent is run in a mock environment seeded with the support graph. Each tool call is recorded along with whether its result matches what the graph said should be there. Mismatches are *defects*:

- **Missing** — a required fact wasn't in the environment.
- **Inconsistent** — the environment provided a value that contradicts a sibling node.
- **Unreachable** — a required tool is not callable in this environment.

### 3. Repair

For each defect, apply a localized fix: inject the missing document, adjust the inconsistent value, or add the missing tool. The *task description* is not changed — only the environment. Re-run the probe; iterate until no defects remain.

### 4. Trajectory generation and SFT

With a repaired environment, run a strong teacher to produce a demonstration trajectory. Pair (task, environment, trajectory) is a training example. Fine-tune a student model via SFT on these pairs.

## Why it matters

- Scales grounded agent-task generation from hundreds (handcrafted) to >100k (automated) across occupations no human curator has seen.
- Probe-and-repair makes the synthetic environment *provably solvable* for the stated intent, which prior synthetic-task pipelines did not guarantee.
- Separates intent (what work to do) from environment (what's needed to do it), opening a general axis for curriculum design: hold intent fixed, vary environment difficulty.

## Gotchas & tricks

- The support graph is only as good as the planner's prompt; occupations outside the planner's training distribution produce under-specified graphs.
- Probes are expensive if the environment is heavy (database + many tools); amortize by running probes in a shared sandbox.
- Diminishing returns on data scale are model-dependent — larger students saturate later than smaller ones.
- Teacher-trajectory SFT is the current recipe; RL on top of these environments is an open direction.
- The intent-fixed, environment-varying setup is a natural pre-stage for RL curriculum work.

## Sources

- Paper: *WorkFactory: Turning Professional Intents into Executable Work for Scalable Agent Training* — Chen, Deng, Wang, Tang, Yang, Ren, Zhao, Chang, et al., ZJU / RUC / SIAT / SJTU / PKU / IQuest Research, 2026 — [arXiv:2610.37161](https://arxiv.org/abs/2610.37161).
- Related: Qwen base models — the backbones used to produce the WorkPilot family.
