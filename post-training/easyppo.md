# EasyPPO
*Depth — a drop-in PPO replacement that fixes two concrete critic failure modes.*

**TL;DR:** EasyPPO (Zhou et al., Berkeley / IBM, 2026) diagnoses PPO's long-standing LLM instability as two **critic** failure modes and fixes each surgically: (i) **actor-only overlong filtering** — filter truncated rollouts from the actor but keep them for the critic — and (ii) **noise-normalized critic regression** — weight each prompt's critic loss by the inverse std of its sampled returns. A moderately smaller critic mini-batch further confines outlier influence. EasyPPO stays stable across coding, math, and multi-turn search benchmarks and beats PPO, VAPO, and HL-Gauss PPO.

**Prereqs:** [ppo.md](./ppo.md), [_rl.md](./_rl.md)
**Related:** [grpo.md](./grpo.md), [rlvr.md](./rlvr.md), [_rewards.md](./_rewards.md)

---

## What it is

Modern LLM-RL made PPO look brittle (hence GRPO's rise). EasyPPO's claim is sharper: PPO's actor is fine; the **critic** is the actual source of collapse. Two specific pathologies and their fixes:

1. **Filtered-truncation drift.** When truncated rollouts are filtered from both actor and critic, the policy objective silently becomes *reward conditioned on completion*. Truncation can rise unboundedly even as the conditional reward improves — the model learns to never finish.
2. **Return-noise heterogeneity.** In finite batches, prompts with high-variance returns dominate the critic gradient. The critic becomes a specialist for the loudest prompts and underfits the quieter ones.

## How it works

### Actor-only overlong filtering

Filter truncated rollouts only when computing the actor loss. The critic learns from returns on **both** completed and truncated trajectories, so it accurately scores incomplete states. This restores the implicit "completion is good" signal.

### Noise-normalized critic regression

For prompt $q$ with sampled returns $r_{q,1} \ldots r_{q,G}$, the per-prompt critic loss is weighted by $1 / \mathrm{std}(r_q)$:

$$
L_{\text{critic}} = \sum_q \frac{1}{\sigma_{r_q}} \sum_i (V(s_{q,i}) - R_{q,i})^2
$$

High-variance prompts (noisy, often outliers) get a smaller weight, so the critic distributes capacity more evenly.

### Smaller critic mini-batches

A moderately smaller critic mini-batch size reduces outlier amplification under PPO's standard gradient-clipping — outliers appear in fewer mini-batches and their influence is confined.

## Why it matters

| Setting | Reward | Benchmark | EasyPPO vs PPO |
|---|---|---|---|
| Continuous | coding | FrontierCS | **+14.89%** |
| Binary | math reasoning | AIME24 | **+2.28%** |
| Multi-turn | search | Search-R1 | **+9.47%** |

EasyPPO stays stable across the full training horizon on all three — vanilla PPO, VAPO, and HL-Gauss PPO all show instability on at least one. The diagnostic (critic as actual culprit) is portable: anywhere an actor-critic LLM method is unstable, check these two failure modes first.

Significance for the field: GRPO was partly motivated by PPO's instability. If PPO's instability was actually a *critic* bug, GRPO's "drop the critic" is one way out but not the only one — EasyPPO re-opens the PPO lane with proper advantage estimates and may regain ground on tasks where a learned value baseline is informative (dense preference rewards, long-horizon agentic RL).

## Gotchas & tricks

- **Actor-only filtering changes advantages.** Returns computed with the critic that saw truncated rollouts are different from returns with a critic that didn't. If you port EasyPPO's recipe, port both halves (filter + noise-normalize) together.
- **Mini-batch sizing interacts with the clip.** Smaller critic batches firing outliers less often means policy-side clip firings also shift. Re-tune $\epsilon$ when adopting.
- **Not a GRPO replacement.** For verifiable-reward reasoning tasks with sparse rewards, GRPO's no-critic design is still the simpler default. EasyPPO wins when a learned value baseline earns its keep — dense rewards, multi-turn, or where completion signal matters.
- **Reward scale across prompts.** Noise-normalization is per-prompt; cross-prompt reward-scale heterogeneity is a separate problem. Pair with reward normalization at the batch level if needed.

## Sources

- Paper: *EasyPPO: Stabilizing the Critic Is Key* — Zhou, Mang, Mao, Li, Chai, Mishra, Wang, Narasimhan, Cheung, Gonzalez — Berkeley / IBM, 2026 — [arXiv:2609.36802](https://arxiv.org/abs/2609.36802).
