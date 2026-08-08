# HarnessOpt-Bench
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A benchmark for LLMs *as harness optimizers*: an optimizer model receives a seed agent harness (prompts, tools, control flow, memory, orchestration code), a graded evaluation with a fixed target-evaluation budget, and must edit the harness and nominate a final candidate. Scoring is the normalized gain over the seed on a held-out test partition that stays inaccessible. Evaluates the meta-capability of improving agent scaffolds, not the base LLM's raw task ability.

**Prereqs:** [_post-training.md](../post-training/_post-training.md) (why post-training matters); [rl-prompt-curation.md](../post-training/rl-prompt-curation.md) (related setting).
**Related:** [../agents/README.md](../agents/README.md) · [../post-training/rl-prompt-curation.md](../post-training/rl-prompt-curation.md) · [README.md](./README.md)

---

## What it is

Modern agent systems are as much about the **harness** — prompt scaffold, tool definitions, control flow, memory, retry logic — as about the model weights. HarnessOpt-Bench turns "how good is a model at improving an agent scaffold" into a measurable capability, isolated from raw task-level ability.

## How it works

**Setup per task.**

1. A **seed harness** for a target agent (prompt + tools + orchestration code) that already runs on the task.
2. A **grading evaluator** the optimizer can call with a fixed budget (noisy, real target-evaluation calls).
3. A **held-out test partition** used only for final scoring — not visible to the optimizer during search.

**Optimizer loop.** An optimizer LLM (paired with a coding harness) iteratively edits the seed. Between edits it can spend its budget calling the graded evaluator on the training partition. Each edit produces a candidate; a trusted execution environment enforces the eval boundary, meters target-agent resource use, and preserves candidate versions for audit.

**Scoring.** The optimizer nominates one final candidate. Its score is the normalized gain over the seed on the held-out partition. Repeating over tasks and seeds yields the leaderboard.

**Two harness axes.** Each optimizer is tested (a) under a shared coding harness across all models — isolates the model — and (b) under its own native harness — measures the whole stack it ships with.

## Why it matters

- **Names a real capability.** In production, most agent quality gains come from harness iteration; this makes those gains measurable.
- **Separates model from scaffold.** The shared-vs-native harness contrast tells you "is this model smart" from "is this vendor's harness well-designed."
- **Budget-explicit.** Fixed-budget graded evaluation matches the constraint every real agent-tuning shop faces.
- Across 5 frontier LLMs and 4 tasks (111 runs), optimizer models separate more than the harnesses they act through — model choice dominates.

## Gotchas & tricks

- **Held-out partition leakage** is the central risk. Sandbox must prevent optimizer from querying the test grader; audit trail must be preserved.
- **Budget setting matters a lot.** Small budget → best-of-N wins; large budget → serious refactoring pays off. Report a curve, not a single number.
- **"Native harness" is not a controlled comparison** — different harnesses expose different tools and different affordances. Interpret only alongside the shared-harness numbers.
- **Task diversity is limited to 4 domains** in the initial release. Extrapolate cautiously; some optimizers may overfit to code-like tasks.

## Sources

- Paper: *HarnessOpt-Bench: Evaluating LLMs at Harness Optimization* — Shanker et al., Scale AI, 2026 — [arXiv:2608.06301](https://arxiv.org/abs/2608.06301).
