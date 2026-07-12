# Unbounded Positive Asymmetric Optimization (UP)
*Depth — an asymmetric clip that unshackles exploration in RL fine-tuning.*

**TL;DR:** Every PPO-family objective used in LLM RL (PPO, GRPO, DAPO, GSPO) clips the importance-sampling ratio symmetrically. UP breaks the symmetry: for *positive* advantages the ratio is left **unclipped** (unbounded exploration for correct but low-confidence trajectories) while for *negative* advantages the standard clip is retained (stability against catastrophic drift). The policy is anchored to its current parameters via a stop-gradient, which is what makes the unbounded positive update numerically safe. Plug-and-play across token-level (GRPO / DAPO) and sequence-level (GSPO) objectives.

**Prereqs:** [ppo.md](ppo.md), [grpo.md](grpo.md), [_rl.md](_rl.md)
**Related:** [rlvr.md](rlvr.md) · [reasoning/long-cot-rl.md](reasoning/long-cot-rl.md) · [rejection-sampling.md](rejection-sampling.md)

---

## What it is

PPO's clipped surrogate limits the ratio $r_{i,t} = \pi_\theta(o_{i,t}\mid\ldots) / \pi_{\theta_\text{old}}(o_{i,t}\mid\ldots)$ to $[1-\epsilon,\, 1+\epsilon]$. UP formalizes the *Probability Capacity* (Cap) — the fraction of the update budget the clip actually spends — and observes that the symmetric clip disproportionately truncates *correct but rare* reasoning trajectories, because a low-probability positive-advantage token has a ratio far above 1 that the clip immediately kills. UP replaces the clip with an asymmetric rule and anchors the policy update with a stop-gradient on the current parameters, preventing runaway divergence.

## How it works

For each rollout, compute the standard advantage $A_{i,t}$ and the ratio $r_{i,t}$. UP applies:

$$
L_\text{UP} = -\frac{1}{G}\sum_i\frac{1}{|o_i|}\sum_t \begin{cases}
\mathrm{sg}(r_{i,t}) \cdot \log \pi_\theta \cdot A_{i,t} & \text{if } A_{i,t} > 0 \\
\min\!\bigl(r_{i,t} A_{i,t},\ \mathrm{clip}(r_{i,t},1-\epsilon,1+\epsilon) A_{i,t}\bigr) & \text{if } A_{i,t} < 0
\end{cases}
$$

where $\mathrm{sg}(\cdot)$ is the stop-gradient. Positive advantages contribute an *unclipped* gradient anchored to the current policy state (bounded by the log-probability's own gradient, which is well-behaved). Negative advantages retain the PPO clip — enough to keep updates that suppress bad trajectories from destabilizing the model. The same rule extends to sequence-level ratios (GSPO): stop-gradient on the sequence-level ratio, unclipped for positive sequence advantages.

## Why it matters

- **Fixes GRPO's exploration ceiling.** In verifiable-reward RL, correct low-probability answers get the highest advantage yet are precisely the trajectories PPO clipping kills. UP is a plug-in fix without new hyperparameters.
- **Universal across variants.** Works with GRPO, DAPO, GSPO; validated on dense, MoE, and vision-language models.
- **No new stability tricks required.** The stop-gradient anchor + asymmetric clip is what buys the unbounded update; there's no schedule to tune.

## Gotchas & tricks

- Stop-gradient on the ratio is essential — remove it and the unclipped positive branch diverges within a few steps.
- The negative-advantage branch keeps standard $\epsilon = 0.2$; the asymmetry is what preserves stability, not a smaller clip.
- Because positive updates are unbounded, entropy can rise unexpectedly early in training — usually helpful for exploration but worth watching in benchmarks that reward determinism.
- Composes cleanly with KL regularization to the reference policy (GRPO's $\beta$ term stays).

## Sources

- Paper: *UP: Unbounded Positive Asymmetric Optimization for Breaking the Exploration-Stability Dilemma* — Fan et al., ByteDance Seed, 2026 — [arXiv:2607.06987](https://arxiv.org/abs/2607.06987).
- Background: *DeepSeekMath (GRPO)* — Shao et al., 2024. See [grpo.md](grpo.md).
- Background: *DAPO* and *GSPO* — see [_rl.md](_rl.md).
