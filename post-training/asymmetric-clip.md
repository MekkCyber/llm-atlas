# Asymmetric Clip (Unbounded Positive)
*Depth — a one-line change to the PPO/GRPO surrogate loss that removes the upper clip on positive advantages.*

**TL;DR:** Standard PPO clips the importance-sampling ratio symmetrically around 1 — the same cap applies whether the token had positive or negative advantage. This throttles exploration on promising trajectories. **Unbounded Positive Asymmetric Optimization (UP)** removes the upper clip so positive-advantage tokens receive unclipped gradients, while keeping the lower clip on negative advantages for stability. Proposed as a drop-in for LLM RL (PPO, GRPO, RLVR, long-CoT-RL) by ByteDance Seed (2026).

**Prereqs:** [ppo.md](./ppo.md), [grpo.md](./grpo.md)
**Related:** [rlvr.md](./rlvr.md), [reasoning/long-cot-rl.md](./reasoning/long-cot-rl.md), [_rl.md](./_rl.md)

---

## What it is

A modification of the PPO clipped surrogate loss that treats positive and negative advantages asymmetrically. Motivated by the observation that in LLM RL, the interesting exploration signal lives in high-positive-advantage tokens (rare good completions), which the symmetric clip caps at ratio $1 + \varepsilon$ — starving the very tokens the policy should push hardest on.

## How it works

PPO's clipped surrogate objective is:

$$
L^{\text{PPO}}(\theta) = \mathbb{E}\left[\min\bigl(r_t(\theta)\, A_t, \;\; \mathrm{clip}(r_t(\theta), 1-\varepsilon, 1+\varepsilon)\, A_t\bigr)\right]
$$

where $r_t(\theta) = \pi_\theta(a_t | s_t) / \pi_{\theta_\text{old}}(a_t | s_t)$ is the importance-sampling ratio.

UP replaces the symmetric clip with:

$$
L^{\text{UP}}(\theta) = \mathbb{E}\left[\begin{cases}
r_t(\theta) \cdot A_t & \text{if } A_t > 0 \\
\min\bigl(r_t(\theta) A_t, \;\mathrm{clip}(r_t(\theta), 1-\varepsilon, \infty)\, A_t\bigr) & \text{if } A_t \le 0
\end{cases}\right]
$$

Positive-advantage tokens: **no upper clip**. Gradients can flow at any magnitude of $r_t$, so a token that turned out unusually good keeps receiving update signal even after the policy shift has already amplified its probability substantially.

Negative-advantage tokens: keep the standard lower clip — this is where instabilities come from (a mistake gets over-punished, the policy collapses), so the safeguard stays in.

The change composes with GRPO (which uses group-relative advantages instead of a value function) and with any RLVR/long-CoT-RL loop that uses a PPO-style surrogate.

## Why it matters

The exploration-vs-stability tradeoff is the dominant tuning axis in modern LLM RL. Every recent open recipe — DeepSeek-R1, Kimi K1.5, DAPO — ships some custom modification of the clip (dual-clip PPO, DAPO's dual thresholds, GRPO's group normalization). UP argues those are all workarounds for the *symmetric* clip and proposes the cleaner fix directly. If it holds up under scrutiny, it becomes a one-line default rather than a per-recipe hyperparameter.

## Gotchas & tricks

- **Only "positive" is asymmetric.** The lower clip on negative advantages is what keeps training stable when the reward model is imperfect — do not remove it.
- **Baseline / advantage normalization still matters.** The trick is about clipping, not about how $A_t$ is estimated. GRPO-style group-relative advantages and PPO-style GAE both compose.
- **Watch reward hacking.** Unbounded positive gradients amplify whatever the reward signal actually is; if the reward model can be gamed, UP will find the game faster than symmetric PPO.
- **KL / entropy regularizer.** With unbounded positive gradients, the KL-to-reference or entropy bonus becomes the effective stabilizer on the positive side — tune it accordingly.

## Sources

- Paper: *UP: Unbounded Positive Asymmetric Optimization for Breaking the Exploration-Stability Dilemma* — Fan et al., ByteDance Seed, 2026 — https://arxiv.org/abs/2607.06987
- Related: *Proximal Policy Optimization Algorithms* — Schulman et al., 2017 — original PPO surrogate.
- Related: *DAPO: an Open-Source LLM Reinforcement Learning System at Scale* — Yu et al., 2025 — dual-threshold clip variant.
