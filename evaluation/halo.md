# HALO — Human-Grounded Granular Evaluation of Proactive Agents
*Depth — a benchmark for proactivity: should the agent help, when, and how far.*

**TL;DR:** HALO is a human-annotated benchmark that measures *proactive* agent behavior along two axes — **timing** (when to intervene) and **intensity** (how far to go). Scenarios span 4 need types × 10 everyday domains; agents retrieve evidence from databases or documents before acting. The release — **HALO-8K** — contains 8,000 responses from 10 models, scored by 20 experts on acceptability and failure cause, plus effort metrics (reasoning tokens, retrieval rounds). Headline failure mode: **over-proactivity** (acting too early or too forcefully). Best proprietary agents reach ~72.9% acceptability; best open-source ~56.6%.

**Prereqs:** [README.md](README.md), [ifeval.md](ifeval.md)
**Related:** [../agents/README.md](../agents/README.md) · [../safety/_scheming.md](../safety/_scheming.md) · [../safety/README.md](../safety/README.md)

---

## What it is

Most agent benchmarks score **execution** — given a task, can the agent complete it. HALO targets a different capability: the **decision to act at all**. For deployed assistants (email, calendar, chat sidekicks) the hard, deployment-blocking question is not "can you do task X" but "should you volunteer to do task X, and if so, how much of it."

HALO operationalizes this with:
- Scenarios that *imply* a possible task but don't request it.
- Human annotations on whether the agent's response was acceptable, and if not, which failure mode it hit.

## How it works

### Scenario design

Four need types × ten domains. The need types describe *why* a user might benefit from a proactive act — e.g. an upcoming calendar conflict, missing context, overlooked follow-up. Each scenario includes:
- A conversation context (text).
- A retrievable corpus (database rows, documents) the agent must consult before acting.
- A ground-truth annotation of what an *acceptable* proactive response would look like.

### Evaluation axes

Two primary axes:
1. **Timing.** Did the agent act too early (over-proactive), too late (passive), or on-time?
2. **Intensity.** Did the agent do too much (hijacking the conversation) or too little (unhelpful hedging)?

Each response is scored for acceptability (binary + reason) and tagged with its dominant failure cause. Expert agreement is reported; the 20-expert panel labels ensures the ground truth is not model-generated.

### Release — HALO-8K

- 8,000 responses from 10 models (mix of proprietary and open).
- Per-response: acceptability label, failure-cause tag, reasoning-token count, retrieval-round count.
- Covers reasoning effort as a secondary variable — enables the "does more reasoning make agents less annoying?" question.

### Headline numbers

- Best proprietary: ~**72.9%** acceptability.
- Best open-source: ~**56.6%** acceptability.
- Dominant failure across all 10 models: **over-proactivity** (acting too early / too forcefully).
- More reasoning tokens does **not** monotonically improve acceptability.

## Why it matters

- **First granular proactivity benchmark.** Prior benchmarks (AgentBench, WebArena) assume the task is given. HALO measures the *gating* step.
- **20-expert human ground truth.** The acceptability axis isn't model-judged; it's expert-labeled, giving credible evaluations for a famously squishy property.
- **A deployment-critical axis.** Over-proactive assistants are the top user complaint in deployed copilots. HALO gives a reproducible benchmark for the fix.
- **A plausible "RewardBench of proactivity."** If adopted, the HALO-8K response pool could become a training set for reward models that score proactivity along the two axes directly.

## Gotchas & tricks

- **Culture-specific norms.** "Appropriate proactivity" is heavily culture- and user-dependent. HALO's panel is one labeling; a different pool could shift the acceptability threshold.
- **Scenarios imply but don't guarantee a right answer.** For some scenarios the acceptable response may be "do nothing." Scoring rewards agents that correctly refuse — a non-trivial behavior to train.
- **Retrieval is in-scope.** An agent that skips retrieval and asserts confidently will often be *less* acceptable than one that admits uncertainty, so retrieval-skipping agents score poorly even if their final suggestion matches ground truth.
- **Reasoning-token spend is non-monotonic.** A direct challenge to "think-more-is-better" narratives for deployed assistants.
- **Small size for its ambitions.** 8,000 responses × 10 models is a modest sample; expect follow-ups that scale annotation.

## Sources

- Paper: *HALO: Human-Grounded Granular Evaluation of Proactive Agents* — Wenqi Pei, Xinyu Wang, Jinrui Liu, Cynthia Zhuge, Jie Sheng, Zihan Wang, Tenghao Huang, Muhao Chen — 2026 — arXiv 2610.39086.
- Related: *IFEval* — Zhou et al., 2023 — the "did you follow the instruction" cousin on the execution axis.
- Related: *AgentBench* — Liu et al., 2023 — agent evaluation on executed tasks.
