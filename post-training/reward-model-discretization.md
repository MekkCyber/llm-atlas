# Reward Model Discretization

*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Continuous neural reward models are systematically **oversensitive** — they assign different scores to responses that are equally good — and the RL policy exploits the noise instead of the signal. Discretization fixes this **without retraining the RM**: run it with Monte Carlo dropout, cluster the resulting per-prompt score distribution into a small number of buckets, and use the bucket label as the reward.

**Prereqs:** [_rewards](_rewards.md), [grpo](grpo.md)
**Related:** [ppo](ppo.md), [cot-reward-model](cot-reward-model.md), [rlvr](rlvr.md)

---

## What it is

A training-free post-processing step that maps a continuous RM's scalar output to a small set of discrete reward levels. Reframes RM quality with two distinct measures:

- **Discriminative ability** — can the RM separate clearly better responses from clearly worse ones?
- **Specificity** (the complement of oversensitivity) — do equally-good responses get the *same* score?

A continuous RM can have perfect discriminative ability while being arbitrarily oversensitive. RL then optimizes the noise, not the intent — a clean theoretical account of one flavor of reward hacking that earlier "RM accuracy" diagnostics miss.

## How it works

For a given prompt $q$, you want a reward function whose level sets are flat over "equally good" responses.

1. **Sample the RM stochastically.** Enable dropout at inference time and score the same $(q, r_i)$ pair $M$ times to get a per-response score distribution.
2. **Cluster across responses.** Across all candidate responses for $q$, cluster the resulting mean-and-variance points into $K$ buckets (e.g., $K=3$ or $K=5$). Methods that work: simple percentile bucketing, k-means on (mean, variance).
3. **Use the bucket label.** Replace the continuous score with the bucket index (or its center). This is the RL reward.

The theorem: there exists a discretization that strictly reduces oversensitivity at minimal cost to discrimination — i.e., the bucket boundaries can be chosen to merge "noise-distinguished" responses while keeping "signal-distinguished" responses separated.

## Why it matters

- **Cleans up a widely-felt RM pathology** that the field had been treating as a generic "reward hacking" symptom.
- **Plugs into existing pipelines.** No RM retraining; works with any neural RM, including big preference RMs and CoT RMs.
- **Reframes the debate** between learned RMs and verifiable rewards: part of the gap is explained by continuous-RM oversensitivity, and the gap can be partially closed by discretization — making preference RMs more competitive on verifiable-style tasks.
- **Naturally meshes with [GRPO](grpo.md):** group-relative advantage estimation already collapses scores within a group; discretizing them first reduces within-group noise that previously drove spurious gradients.

## Gotchas & tricks

- **Bucket count is a hyperparameter.** Too few → loss of useful discrimination; too many → recover continuous-RM oversensitivity. The paper sweeps and recommends small $K$.
- **MC-dropout variance is a property of the RM, not the response.** Some RMs have near-deterministic outputs even with dropout enabled; on those, discretization helps less. Calibration check before deploying.
- **Compatible with off-policy training** because it's a function on RM outputs, not a change to the RL update.
- **Doesn't fix label-noise** in the RM training data — it only smooths over the RM's own response-level oversensitivity. Discretization on a fundamentally miscalibrated RM still misranks.
- **Generative RMs are already partly discrete** (the CoT RM emits a JSON judgment with discrete fields). This paper rationalizes that design choice as the right default for any RL recipe.

## Sources

- Paper: *Discretizing Reward Models* — Viswanathan, Wang, Hazarika, Nagpal, Wu, Neubig, Mao, 2026 — [arXiv:2606.21795](https://arxiv.org/abs/2606.21795) — CMU / Meta Superintelligence Labs.
- Background: *Generative Verifiers: Reward Modeling as Next-Token Prediction* — Zhang et al., 2024 — generative RMs that are intrinsically near-discrete.
- Background: *DeepSeek-R1* — DeepSeek, 2025 — explicit choice of rule-verifier rewards in part to avoid this failure mode.
