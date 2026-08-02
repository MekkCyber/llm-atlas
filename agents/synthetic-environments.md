# Synthetic Environments for Agent Training
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Computer-use agents need applications they can *act on, break, and reset* — but the real ones that matter (Salesforce, Jira, hospital EHRs) are login-gated and stateful. Synthetic environments compile a **specification into a stateful application** with graded tasks against the app's own database, then **co-evolve** with the agent: each rollout is read twice, once as a repair to the environment and once as training signal. Environment quality — depth, targeting, evolution — turns out to matter more than raw environment count.

**Prereqs:** [rlvr](../post-training/rlvr.md)
**Related:** [rl-prompt-curation](../post-training/rl-prompt-curation.md), [_rl](../post-training/_rl.md)

---

## What it is

The bottleneck for training computer-use agents has moved. Pipelines that generate synthetic environments in bulk have solved the *how many* problem. What matters now is *what's inside each one*:

- **Behavioural depth** — enough branching state, edge cases, and cross-view consistency that the agent has room to fail in different ways.
- **Targeting** — the environment surfaces exactly the interaction pattern the agent currently fails.
- **Co-evolution** — the environment (its tasks, its verifier) improves alongside the model instead of saturating.

An environment is a compiled specification: a stateful application backed by its own database, plus tasks graded by SQL/database checks against that state. This makes rewards verifiable at any point in a rollout — the environment *is* the verifier.

## How it works

Echoverse (Microsoft Research, 2026) instantiates this pattern:

1. **Compilation.** Author a spec (entities, actions, invariants). The compiler emits a runnable web/desktop application with a live database. Grading is expressed against the database schema.
2. **Rollout & grading.** The agent acts; the verifier reads the database state and emits a graded trajectory.
3. **Co-evolution loop.** Each graded rollout feeds two consumers:
   - **Environment side:** LLM-authored repairs to the environment definition, tasks, and verifier (fixes ambiguous instructions, flaky invariants, unreachable states).
   - **Agent side:** trajectory becomes training data (SFT + RL).

Depth beats breadth: drilling one interface control across many renderings **transfers to held-out widget families and to the open web** — a 9B model trained on 12 co-evolving environments improved from 36.5% to 67.1% across 14 evaluation splits, within 14 points of its much larger teacher.

## Why it matters

- **Verifiable rewards for computer use.** Grading against the app's own database gives every rollout a clean, dense reward signal — RLVR for GUI agents.
- **Environment quality > environment count.** Shallow synthetic environments *reduce* live-site accuracy (80.0 → 75.0); deep ones raise it (80.0 → 85.0, 48.0 → 65.0). Repairing one environment lifts its agent from 16.2% to 38.5%.
- **RL and SFT reuse the same worlds.** A grounded verifier plus a per-step judge gives a dense reward that raises held-out score from 58.8% to 68.0% under RL — no separate RL infrastructure needed.

## Gotchas & tricks

- **Depth is the hidden lever.** Bulk environments are easy; deep environments — those that carry real state branching, cross-view consistency, and failure paths worth training on — are expensive. Skip depth and you may actively *regress* on real-world benchmarks.
- **Verifier drift is real.** As the agent learns to game the grader, the grader must be repaired. This is what the "co" in co-evolution does.
- **Login-gated / stateful surrogates.** The whole point is applications *you* can reset. Synthetic environments are how you get training signal on categories (finance, healthcare, enterprise) that real APIs won't give you.
- **Not a replacement for real-site eval.** Use synthetic environments for training and initial eval; keep a live-site or held-out benchmark to catch spec-vs-reality drift.

## Sources

- Paper: *Echoverse: Deep, Evolving Environments for Training Computer-Use Agents at Scale* — Pandya, Gupta, Harne et al. (Microsoft Research), 2026 — [arXiv:2607.28074](https://arxiv.org/abs/2607.28074).
