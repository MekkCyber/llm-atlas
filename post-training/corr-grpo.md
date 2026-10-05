# CorrGRPO — Correlation-Normalized GRPO for Multi-Reward Learning
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** In multi-reward GRPO, the within-group variance equals the sum of all pairwise reward covariances — so correlated large-scale rewards dominate the normalisation and suppress smaller-scale ones. CorrGRPO replaces each pairwise covariance with a Pearson correlation coefficient, keeping the centred total reward unchanged but rebalancing contributions by scale.

**Prereqs:** [grpo.md](./grpo.md), [_rewards.md](./_rewards.md)
**Related:** [dara.md](./dara.md), [rlvr.md](./rlvr.md)

---

## What it is

Hu et al. (HKUST, 2026) analyse a specific failure mode of multi-reward GRPO. GRPO normalises the centred total reward by its within-group standard deviation; for a sum of rewards, the variance equals the sum of pairwise covariances. CorrGRPO replaces those covariances with Pearson correlations so that scale differences across rewards no longer dominate the normalisation.

## How it works

**Standard GRPO (multi-reward).** For each prompt, roll out $G$ responses, compute a scalar reward per response (sum of components), centre by the group mean, divide by the within-group std. Advantages are then used in the clipped PPO update.

**The imbalance.** $\mathrm{Var}(\sum_i r_i) = \sum_{i,j} \mathrm{Cov}(r_i, r_j)$. A reward with large magnitude *and* positive correlation with other rewards inflates the group std, shrinking all advantages — including those of smaller rewards that may be the ones actually providing new signal.

**The fix.** Rewrite the pairwise contributions as Pearson correlations:
$$\rho_{ij} = \frac{\mathrm{Cov}(r_i, r_j)}{\sigma_{r_i} \sigma_{r_j}}$$
and use these correlation coefficients (scale-invariant, in $[-1, 1]$) in the normalisation. The centred total reward itself is not changed — only the normaliser. The result: advantage magnitudes still adapt to reward *dependence* across the group, but can no longer be dominated by a single large-scale reward.

## Why it matters

- Direct fix for the common "format reward on 0/1 scale vs length reward on raw-token scale" pathology — any multi-reward recipe with mixed scales benefits.
- Reported gains across three settings: code generation, tool calling, agent security — on model sizes from 0.5B to 8B.
- Lands alongside [DARA](./dara.md), which attacks the orthogonal sparsity axis. Together they fill out GRPO's multi-reward branch: DARA for density, CorrGRPO for scale/correlation.
- Code at [github.com/HKUST-KnowComp/CorrGRPO](https://github.com/HKUST-KnowComp/CorrGRPO).

## Gotchas & tricks

- Correlation estimates are noisy when the group size $G$ is small; the paper's recipe assumes a $G$ large enough for stable pairwise estimates.
- A reward with zero within-group variance (every rollout scored the same) still breaks the correlation computation — fall back to the centred-mean advantage with no normalisation, same as the standard GRPO edge case.
- Does not help when rewards are correlated *and* dense *and* same-scale (nothing to rebalance) — benefit depends on there being an imbalance to fix.
- Composes cleanly with DARA — apply density correction first, correlation normalisation second.

## Sources

- Paper: *CorrGRPO: Correlation-Normalized GRPO for Multi-Reward Learning* — Hu, Jing, Shi, Liu, Li, Song; HKUST, 2026 — [arXiv:2609.36820](https://arxiv.org/abs/2609.36820).
- Code: [https://github.com/HKUST-KnowComp/CorrGRPO](https://github.com/HKUST-KnowComp/CorrGRPO).
- Context: *GRPO* — Shao et al., DeepSeekMath, 2024 — the base algorithm this variant modifies.
