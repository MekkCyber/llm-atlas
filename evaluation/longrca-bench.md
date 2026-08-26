# LongRCA-Bench
*Depth — 1,140 real long-horizon agent failures, scored on responsible role and earliest decisive root-cause step.*

**TL;DR:** Most agent-failure benchmarks work on short traces. LongRCA-Bench covers 1,140 real failed trajectories (median 145 steps) across five domains, each labeled by humans with two independent targets: the *responsible role* (which agent/tool/user turn is at fault) and the *earliest decisive root-cause step* (exact index in the trace). It also introduces RCTA, a training-free retrieval-based localizer.

**Prereqs:** [../agents/rlm-harness](../agents/rlm-harness.md)
**Related:** [mobilepa-bench](mobilepa-bench.md)

---

## What it is

A diagnostic benchmark for post-mortem analysis of long-horizon agent runs. Given a failed trajectory (potentially hundreds of steps), the benchmark asks the evaluator to identify **which role failed** and **at which exact step the decisive error was made**. Human labels for both.

## How it works

- **Data:** 1,140 failed trajectories collected from real runs (not synthetic error injection) across five domains. Median trajectory length: 145 steps.
- **Labels (independently scored):**
  - *Responsible role* — which participant (planner, sub-agent, tool, user turn) carried the decisive error.
  - *Earliest decisive root-cause step* — the exact trajectory index where the failure became inevitable.
- **Baselines:** frontier LLMs applied to the full transcript reach only ~13.2% exact-root-step accuracy. Responsible role is easier but still noisy.
- **RCTA (Root-Cause Trajectory Attribution).** Training-free method:
  1. Segment the trajectory into summarizable chunks.
  2. Retrieve candidate error steps from segment summaries (embedding search).
  3. Trace each candidate back to earlier handoff instructions to find where the deviation started.
  4. Return the earliest candidate whose upstream instruction was reasonable → that's the decisive step.
  Reaches 51.1% responsible-role and 24.1% exact-root-step accuracy on the same backbone.

## Why it matters

Outcome-level evaluation tells you a run failed; it does not tell you *where*. Without exact root-step localization you can only iterate on outcomes, not on causes. LongRCA-Bench makes root-cause localization a first-class benchmark target and shows meaningful headroom on it.

## Gotchas & tricks

- Splitting responsible-role and root-step into two independently scored targets is important — a good role predictor can still be far off on the exact step.
- Errors are *real* (not injected), which is a strength and a limitation: the label distribution reflects what actually happens in production, not what stress-testing would surface.
- RCTA is training-free and thus a strong baseline for methods that require fine-tuning to justify their cost.

## Sources

- Paper: *LongRCA Bench: Diagnosing Responsible Roles and Root Causes in Long-Horizon Agent Failures* — Zhang et al., 2026 — [arXiv:2608.15242](https://arxiv.org/abs/2608.15242)
