# Best-of-N (BoN)
*Depth — inference-time scaling by sampling many candidates and picking the highest-scoring one.*

**TL;DR:** Generate $N$ independent candidates for a prompt, score each with a verifier or reward model, and return the best. The simplest inference-time scaling method. Its cost is $N \times$ generation + $N \times$ scoring; its quality tracks the reward-model / verifier quality closely. Modern variants (Flash-BoN, guided search, tree-of-thought BoN) reduce the effective cost by making the drafts cheap.

**Prereqs:** [../post-training/reasoning/orm.md](../post-training/reasoning/orm.md), [../post-training/rejection-sampling.md](../post-training/rejection-sampling.md)
**Related:** [../post-training/_rewards.md](../post-training/_rewards.md), [../post-training/reasoning/prm.md](../post-training/reasoning/prm.md), [../post-training/reasoning/mcts.md](../post-training/reasoning/mcts.md)

---

## What it is

Given a prompt $x$, a generator $\pi$, and a scorer $R$:

1. Sample $N$ candidates $\{y_1, \ldots, y_N\} \sim \pi(\cdot | x)$.
2. Score each: $s_i = R(x, y_i)$.
3. Return $y^\star = \arg\max_i s_i$.

Applies to both LLM outputs (with an outcome-reward model or verifier) and diffusion outputs (with a preference model or CLIP-style scorer). The **cheapest** inference-time-scaling family — no per-step tree search, no rollout planner. Also the **highest-variance** — because $R$ must actually be trustworthy for BoN to help.

## How it works

The core loop is trivial; the interesting knobs are:

- **Draft cost.** Naïve BoN samples $N$ full completions at full quality. Modern variants make the drafts cheap:
  - **Timestep truncation** (diffusion): stop early, ranked by an early-step score.
  - **Layer skipping**: run a subset of transformer or DiT layers per draft.
  - **Activation proxies**: predict quality from an intermediate representation rather than a full completion.
  - **Draft models** (LLMs): a smaller "draft" model produces candidates; the target model only verifies.
- **Score cost.** For expensive verifiers (LLM-as-judge, process reward), a staged filter first (quick score, top-K survive) then a stronger scorer refines. This is Flash-BoN's multi-stage verification.
- **Sampling diversity.** Independent samples at temperature $T > 0$ tend to cluster; nucleus sampling, prompt variation, or explicit diversity penalties broaden coverage of the $N$ candidates.

The **weak-verifier failure mode** is BoN's central risk: as $N$ grows, BoN over-optimizes whatever the reward model rewards. If the reward is a proxy, BoN maximizes the proxy, not the true objective.

## Why it matters

- **Simplest inference-time scaling that works.** No search algorithm, no policy training — just samples and a scorer. This is why RLHF-with-BoN was the first widely-adopted inference-time compute lever (WebGPT, Anthropic's early assistants).
- **Wall-clock beats NFE.** Recent diffusion work (Flash-BoN) argues that when compared under a **wall-clock budget** — not a fixed number of function evaluations — cheap-draft BoN matches or beats guided intermediate-step search. Serving-time metrics should be wall-clock.
- **Composes with prompt optimization and RL.** Reflection-style prompt rewriting can precede BoN; RL post-training can be initialized from BoN traces (accelerates convergence).

## Gotchas & tricks

- **Reward hacking scales with $N$.** BoN with an imperfect reward model produces monotonically worse outputs past some $N^\star$ — reward-model quality is the binding constraint.
- **Draft cheapness matters more than $N$.** Doubling $N$ helps less than halving draft cost — the wall-clock frontier is dominated by draft-per-second, not by the theoretical $N$.
- **Verifier caching.** For LLM tasks, cache the KV of the shared prompt across the $N$ candidates. For diffusion, cache the shared encoder passes.
- **Compare wall-clock, not NFE.** NFE-matched BoN can look bad against methods that squeeze more work per NFE; wall-clock is the honest metric.
- **BoN → RL flywheel.** Traces produced by BoN can seed rejection sampling or GRPO / RLVR training — a common post-training loop.

## Sources

- Paper: *Flash-BoN: Instant Drafts for Inference-Time Scaling in Diffusion Models* — Shirkavand et al., UMD / Hugging Face, 2026 — https://arxiv.org/abs/2607.04461 — the cheap-draft wall-clock argument.
- Paper: *WebGPT: Browser-assisted question-answering with human feedback* — Nakano et al., 2021 — early BoN-with-RLHF at scale.
- Paper: *Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters* — Snell et al., 2024 — inference-time compute vs parameter scaling.
- Paper: *Are Emergent Abilities of Large Language Models a Mirage?* — Schaeffer et al., 2023 — reward-hacking failure modes at large $N$.
