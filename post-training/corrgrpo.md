# CorrGRPO — Correlation-Normalized GRPO for Multi-Reward Learning
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A GRPO variant for multi-reward RL. Standard GRPO sums reward components and normalizes the total by group std, which lets a large-scale, highly-correlated reward dominate smaller-scale signals. CorrGRPO normalizes *pairwise reward covariances* into Pearson correlation coefficients before summing, so each reward's contribution reflects its information content, not its raw magnitude. Preserves the centered total reward and plugs into existing GRPO code with one function swap.

**Prereqs:** [grpo.md](grpo.md), [_rewards.md](_rewards.md)
**Related:** [ppo.md](ppo.md) · [_rl.md](_rl.md) · [rlvr.md](rlvr.md)

---

## What it is

Modern agentic RL stacks three to six reward terms per rollout: correctness, format, safety, tool-use validity, style, length. Standard GRPO (see [grpo.md](grpo.md)) sums them to a scalar $r_i = \sum_j r_{i,j}$ and uses the per-group z-score as advantage:

$$
A_i = (r_i - \bar{r}) / \sigma_r
$$

Two failure modes with this:

1. **Scale dominance.** If one reward's within-group std is 10× another's, it dominates the sum and the smaller signal is effectively a tiebreaker.
2. **Correlation leakage.** If two rewards are highly correlated (e.g. "answer correct" and "answer format valid"), they double-count that direction.

CorrGRPO fixes both by moving to the correlation matrix.

## How it works

For each prompt's group of $G$ rollouts:

1. Compute the per-reward vectors $r_{:,j}$ across the $G$ rollouts.
2. Compute pairwise sample covariances $\mathrm{Cov}(r_{:,j}, r_{:,k})$.
3. Convert to Pearson correlations $\rho_{jk}$ by dividing by the product of per-reward stds.
4. Reweight the summed reward's variance using the correlation matrix: the effective std is derived from the sum of weighted covariances (correlations) instead of the raw-sum std.
5. Form $A_i$ as the centered sum normalized by this effective std; the group-centred total reward is preserved.
6. Feed $A_i$ into the standard PPO-clipped update (GRPO's existing surrogate loss).

Only steps 2–4 are new. Rollouts, loss, KL penalty are all identical to GRPO.

## Why it matters

- **Prevents one metric eating the others.** Especially useful on agent-safety RL where correctness and safety are correlated — their joint information is less than their sum of scales.
- **Compatible with existing recipes.** No new network, no new hyperparameter you must tune (correlation estimates are from the group itself).
- **Measured gains.** The paper reports improvements over baseline GRPO on code generation, tool-calling, and agent-security tasks for 0.5B–8B models.

## Gotchas & tricks

- **Needs reasonable $G$.** The correlation matrix is noisy at $G=4$; the paper recommends $G \ge 8$, matching GRPO practice.
- **Degenerate correlations.** If a reward is constant across the group, $\sigma = 0$ and $\rho$ is undefined. Clip or skip; also skip the whole reward from the sum since it's uninformative.
- **Not a replacement for scalarization choice.** You still have to decide which rewards to include and whether to clip or shape them; CorrGRPO only fixes the within-group normalization.
- **Covariance is per-prompt, not global.** Each group gets its own correlation matrix; this is intentional (reward relationships vary by task) but means low-$G$ noise matters.

## Sources

- Paper: *CorrGRPO: Correlation-Normalized GRPO for Multi-Reward Learning* — Hu, Jing, Shi, Liu, Li, Song; HKUST, 2026 — [arXiv:2609.36820](https://arxiv.org/abs/2609.36820).
- Related: GRPO (Shao et al., 2024, DeepSeekMath) — the baseline whose normalizer CorrGRPO swaps.
