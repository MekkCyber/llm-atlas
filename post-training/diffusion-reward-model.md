# Diffusion Reward Model (DRM)
*Depth — reward modeling as conditional density estimation over p(r | x, y) using a lightweight Diffusion Transformer.*

**TL;DR:** Standard reward models compress each (prompt, response) into a single scalar or a fixed parametric distribution. DRM instead trains a Diffusion Transformer to denoise Gaussian noise into a reward vector conditioned on a frozen LLM encoder, giving a non-parametric density that naturally represents multimodal human preference.

**Prereqs:** [_rewards](_rewards.md), [cot-reward-model](cot-reward-model.md)
**Related:** [dpo](dpo.md), [grpo](grpo.md), [ppo](ppo.md)

---

## What it is

A reward-model head that replaces the usual "scalar via MLP" or "Gaussian via mean+variance heads" with a small Diffusion Transformer (DiT). The DiT is conditioned on the frozen LLM encoder's representation of `(x, y)` and learns `p(r | x, y)` as a distribution rather than a point.

## How it works

- **Encoder.** A frozen instruct/base LLM produces the representation `h(x, y)` used as diffusion conditioning.
- **Diffusion head.** A lightweight DiT denoises reward-space Gaussian noise into a reward vector, trained with the standard diffusion loss on paired (or scored) preference data. No parametric family is imposed on the output distribution.
- **Consumption downstream.** The RM emits a *sample* per call (or a small batch of samples for quantile estimates). Policy-optimization algorithms consume the mean, a chosen quantile, or an uncertainty-aware bonus derived from sample variance.

## Why it matters

Human preference is genuinely multimodal: the same response can be judged in incompatible-but-reasonable ways. Scalar RMs collapse that structure; parametric-Gaussian RMs impose a shape that doesn't cover bimodal cases. DRM keeps the multimodality, which (a) reduces reward-hacking pressure — the policy has to defend against a distribution rather than a single score — and (b) opens a clean path to quantile-based PPO/GRPO variants (train against a lower quantile for risk-averse alignment). Reported gains on RM benchmarks match or beat scalar/parametric baselines, with the biggest wins on prompts where preference is legitimately ambiguous.

## Gotchas & tricks

- **Sampling cost.** Each RM call is now a mini diffusion sample. In PPO/GRPO rollouts this multiplies RM latency; a small denoising-step budget (e.g. 4–8 steps) usually suffices for stable training.
- **Encoder frozen.** Because the encoder is frozen, DRM captures preference density but not new representation learning. If encoder features are already saturated (small base model, narrow prompt distribution), DRM won't beat a scalar RM.
- **Quantile choice matters.** Using the mean recovers approximately-scalar behavior; the payoff comes from picking a quantile appropriate for the alignment goal (a low quantile for cautious deployment, the mean for standard PPO).
- **Reward-model overfitting** is easier to hide when the output is a distribution — always check calibration on held-out preference pairs, not just win rate.

## Sources

- Paper: Diffusion Reward Models — Wang, He, Liu, Wang, Qiao, Zuo, Gao, Qian, et al., 2026 — [arXiv:2609.33803](https://arxiv.org/abs/2609.33803)
