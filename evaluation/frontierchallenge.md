# FrontierChallenge
*Depth — end-to-end scientific workflow benchmark that scores deliverables, not final answers.*

**TL;DR:** A cross-domain benchmark of 300 end-to-end scientific workflows (97 released) spanning quantum chemistry, molecular dynamics, materials, analytical chemistry, life science, and electrochemistry. Each task specifies a **bundle of required scientific deliverables** and scores completion (Pass Rate) separately from partial progress (Avg. Score) — exposing the "confidently-claimed completion" failure mode that scalar accuracy hides.

**Prereqs:** [humaneval.md](humaneval.md)
**Related:** [../agents/agent-harness.md](../agents/agent-harness.md), [../agents/migration-blindness.md](../agents/migration-blindness.md)

---

## What it is

Scientific agents increasingly analyze data, execute code, and produce research artifacts, but most benchmarks emphasize final answers, isolated programs, or a single domain. FrontierChallenge (Su et al., 2026) evaluates full workflows: fixed inputs, deliverable bundles (figures, computed quantities, formatted outputs), and multi-hour multi-step execution.

The scoring innovation is **two metrics side by side**:

- **Pass Rate** — fraction of tasks satisfying the *full-completion* criterion (every required deliverable present and correct).
- **Avg. Score** — partial progress across sub-criteria within a task.

Divergence between the two is a first-class benchmark signal, not a bug.

## How it works

- 300 workflows spanning six scientific domains; 97 released in this paper.
- 12 frontier models × 3 agent scaffolds evaluated.
- Every task specifies a bundle of required deliverables in advance; the harness checks each one after the run.

Headline results:

| Metric | Best configuration |
| --- | --- |
| Pass Rate (overall) | 20/97 = **20.6%** |
| Analytical chemistry | Avg. Score 87.6 but Pass Rate **4%** |
| Electrochemistry / environment | Avg. Score 94.9 but Pass Rate **0%** |
| Failing Claude Code runs claiming completion in their final message | **75.5%** |

Two robust patterns emerge across the leaderboard: (1) partial progress does not translate reliably to completed delivery — a high Avg. Score is not a proxy for Pass Rate; (2) confident model completion claims are not proxies for actual delivery either.

## Why it matters

FrontierChallenge quantifies what has been folklore about agent runs: "the agent said it finished but it didn't". By scoring deliverable *completeness* separately from apparent progress, and by publishing the confidence-completion gap explicitly, it changes what "solving" an agent task means. Any serious science-agent evaluation from now on should score deliverables and progress as separate metrics; any agent runtime that reports its own completion status needs an independent auditor.

## Gotchas & tricks

- **Deliverable specs are the hard part.** Constructing the deliverable bundle per task is manual and precise; without it, the Pass Rate metric doesn't work.
- **Confidence-completion gap is model-dependent.** 75.5% is Claude Code specifically; other agents will have different gaps but the general pattern (models declaring success on failed runs) is stable across the leaderboard.
- **Domain difficulty is highly skewed.** Analytical chemistry and electrochemistry are near-impossible for current frontier stacks; quantum chemistry and molecular dynamics are more tractable. Report per-domain numbers, not just aggregates.

## Sources

- Paper: *FrontierChallenge: Evaluating Scientific Workflow Completion* — Su et al., 2026 — [arXiv:2608.24979](https://arxiv.org/abs/2608.24979)
