# SA-MRPO: Saturation-Aware Multi-Reward Policy Optimization
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Standard multi-reward RL post-training scalarizes rewards with a **fixed weighted sum** *before* group-wise standardization (as in GRPO). Two failure modes fall out: (1) rollouts with distinct reward profiles can end up with identical advantages, erasing useful signal, and (2) gradient budget keeps flowing into already-saturated objectives. **SA-MRPO** (Chen et al., 2026) standardizes each reward objective **independently** and adaptively **down-weights saturated objectives** based on batch-level statistics, so optimization keeps pushing the rewards that still have headroom.

**Prereqs:** [grpo.md](grpo.md), [_rl.md](_rl.md), [_rewards.md](_rewards.md)
**Related:** [rlvr.md](rlvr.md) · [reasoning/long-cot-rl.md](reasoning/long-cot-rl.md)

---

## What it is

Almost every modern reasoning-RL pipeline is now multi-reward: accuracy + format + length + language-consistency + safety, stacked. The default combination is *scalarize then normalize*:

$$
r_i = \sum_k w_k \cdot r_i^{(k)}, \qquad A_i = \frac{r_i - \bar r}{\sigma_r}
$$

SA-MRPO argues this loses information twice: distinct reward profiles get flattened into one scalar before the baseline, and the fixed weights $w_k$ never reflect how close each objective is to its ceiling.

## How it works

Two edits to the GRPO advantage estimator:

1. **Per-objective group-wise standardization.** For each objective $k$, compute the group mean and std across the $G$ rollouts of a prompt and z-score independently:

    $$
    A_i^{(k)} = \frac{r_i^{(k)} - \bar r^{(k)}}{\sigma_r^{(k)}}
    $$

    Two rollouts with different reward profiles now get different advantages, per objective.

2. **Saturation-aware reweighting.** Estimate per-objective saturation at the batch level — how close the reward is to its known or empirical ceiling — and use it to down-weight the objective's contribution:

    $$
    A_i = \sum_k \alpha_k(\text{sat}^{(k)}) \cdot A_i^{(k)}
    $$

    where $\alpha_k$ decreases as objective $k$ saturates. Solved objectives fade out of the gradient; under-optimized ones dominate.

Otherwise the PPO-clipped update and KL-to-reference term are unchanged — SA-MRPO is a drop-in advantage change with no rollout-cost overhead.

## Why it matters

- Multi-reward RL is now the default for reasoning + safety + format + code post-training. If the community keeps stacking rewards, *which advantage estimator* matters more than *which base RL algorithm*.
- Fixes a real pathology: without saturation-aware reweighting, an already-satisfied "format" reward keeps consuming gradient budget while a still-under-target "correctness" reward gets underweighted.
- Compatible with GRPO / mirror-descent / any policy-gradient loop that already computes group statistics — no infrastructure change.

## Gotchas & tricks

- **Saturation estimation is the tricky bit.** For rule-based binary rewards, "saturated" is close to "batch pass rate ≈ 1." For scored rewards, you need a per-objective ceiling — either declared (bounded reward) or estimated online.
- **Under-optimized ≠ hard.** An objective can look under-optimized because the reward function itself is broken (unreachable or badly shaped). Naively pouring more gradient at it is unhelpful; monitor per-objective reward *trajectories*, not just the levels.
- **Compatibility with entropy-regularized RL.** The per-objective standardization changes the effective KL to the reference, so the $\beta$ that worked for scalarized GRPO likely needs re-tuning.
- **Group size $G$ still matters.** Per-objective z-scores need enough samples per objective per prompt to be meaningful; the $G = 8$ that works for scalarized GRPO is probably too small here.

## Sources

- Paper: *Learn What's Left, Not What's Mastered: Saturation Aware Advantage Reweighting for Multi-Reward Policy Optimization* — Yifei Chen, Haichao Zhang, Haozheng Luo, Xander Wu, Jie Ni, Yun Fu, Nuno Vasconcelos, Yijiang Li — arXiv:2608.16072 — 2026.
- Related: *DeepSeekMath* (introduces GRPO) and *DeepSeek-R1* (composite rewards summed before group standardization — the exact pattern SA-MRPO targets).
