# RL trust regions for LLMs

*Taxonomy — how PPO-family RL algorithms keep the new policy close to the rollout policy.*

**TL;DR:** Every PPO/GRPO-style policy update needs a *trust region* — a constraint that stops the new policy from drifting so far from the rollout policy that the importance-weighted gradient becomes meaningless. The classical answer is PPO's symmetric ratio clip $[1-\epsilon, 1+\epsilon]$, applied uniformly to every token. Recent work breaks that answer along three orthogonal axes: **what** the constraint measures (ratio vs. exact divergence), **how widely** it applies (uniform vs. per-token), and **how sharply** it enforces (hard mask vs. smooth penalty).

**Related taxonomies:** [_rl](_rl.md) · [_rewards](_rewards.md)
**Depth files covered here:** [ppo](ppo.md) · [grpo](grpo.md) · [token-level-trust-region](token-level-trust-region.md) · [flow-rl](flow-rl.md)

---

## The problem

PPO/GRPO are off-policy after the first inner gradient step: the policy you're updating ($\pi_\theta$) is no longer the policy that generated the rollout ($\pi_{\theta_{\text{old}}}$). The importance-weighted objective is only a reliable surrogate for the true return *near* $\pi_{\theta_{\text{old}}}$. The trust region exists to keep us there. Get it wrong and one of two failures happens: too tight → policy can't move and training stalls; too loose → policy overshoots, rollouts go off-distribution, training collapses.

## The shared pattern

Every trust-region scheme has three pieces:
1. A **measurement** — what counts as "drift" (ratio $\pi_\theta/\pi_{\theta_{\text{old}}}$, KL, total variation, etc.).
2. A **scope** — applied per token, per response, or globally.
3. An **enforcement** — clip the gradient, mask it out, or scale it down smoothly.

## Variants

| Technique | Measurement | Scope | Enforcement | When it wins |
| --- | --- | --- | --- | --- |
| [**PPO clip**](ppo.md) (Schulman 2017) | ratio | per-token, uniform $\epsilon$ | hard min-clip | classical baseline; default for RLHF |
| [**GRPO clip**](grpo.md) (DeepSeekMath 2024) | ratio | per-token, uniform $\epsilon$ | hard min-clip, group-mean baseline | reasoning RL / RLVR |
| [**DAPO**](#) (Yu 2025) — *no depth file yet* | ratio | per-token, **asymmetric** $\epsilon_\text{low}, \epsilon_\text{high}$ | hard | long-CoT exploration (positive-advantage ceiling raised) |
| [**CPPO**](token-level-trust-region.md) (Tencent Hunyuan 2026) | ratio | per-token, **context-dependent** $\epsilon$ | hard | long-CoT with mixed token entropy |
| **DPPO** (precursor to DRPO) | binary total variation | per-token | **hard mask** (zero gradient outside) | smoother than clip near edges; sharper at threshold |
| **DRPO** (Tencent Hunyuan 2026) — *no depth file yet* | binary TV | per-token | **smooth penalty** weighted by divergence | DPPO with continuous gradients |
| [**Flow-DPPO**](flow-rl.md) (Tencent Hunyuan 2026) | **exact KL** (Gaussian closed-form) | per denoising step | **asymmetric mask** (block only harmful directions) | RL for flow-matching / diffusion |

## How to choose

- **Reasoning RL on text LLMs:** GRPO clip remains the default. CPPO is a drop-in if you're already seeing uniform-clip pathologies in long CoTs (gradient zero on most reasoning forks). DAPO's asymmetric clip is the simpler version of the same idea.
- **Flow / diffusion RL:** the ratio estimate is one-sample-noisy because per-step policies are Gaussian — use Flow-DPPO's closed-form KL. Don't import the text-PPO clip unchanged.
- **Stability vs. final reward:** smooth schemes (DRPO) optimize more gently than hard masks; they help when training is unstable but typically reach similar peaks if the hard scheme converges.
- All of these compose with the **global** KL-to-reference penalty (separate from the trust region). The trust region is intra-iteration; the KL-to-ref is across-iteration drift control.

## Adjacent but distinct

- [**DPO**](dpo.md) — no rollouts, no trust region. The KL-to-reference is baked into the closed-form preference loss.
- **TRPO** — second-order trust region with a Fisher-vector product; the algorithm PPO replaced.
- **Mirror descent ([k1.5](reasoning/online-policy-mirror-descent.md))** — uses an $\ell_2$ surrogate on log policy ratios, which is a smooth trust region by construction; no clip required.

## Sources

- *Proximal Policy Optimization Algorithms* — Schulman et al., 2017.
- *DeepSeekMath* — Shao et al., 2024 — GRPO.
- *DAPO* — Yu et al., 2025 — asymmetric clip for long-CoT exploration.
- *Beyond Uniform Token-Level Trust Region in LLM Reinforcement Learning* — Mao et al., Tencent Hunyuan, 2026 — CPPO.
- *Rethinking the Divergence Regularization in LLM RL* — Yao et al., Tencent Hunyuan, 2026 — DRPO.
- *Flow-DPPO* — Ping et al., XJTU / Tencent Hunyuan / NUS, 2026.
