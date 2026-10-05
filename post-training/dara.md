# DARA — Density-Aware Reward Aggregation for Multi-Reward RL
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** In multi-reward GRPO training, rewards that fire sparsely (produce nonzero relative advantage in only a few rollout groups per batch) carry less learning signal than frequently-active rewards, even after per-reward normalisation. DARA weights advantages by an inverse-square-root density correction derived from each batch's active-group fraction — no change to the policy objective, big speedup on compliance-style rewards.

**Prereqs:** [grpo.md](./grpo.md), [_rewards.md](./_rewards.md)
**Related:** [corr-grpo.md](./corr-grpo.md), [rlvr.md](./rlvr.md)

---

## What it is

Zheng et al. (2026) study multi-reward RL post-training with GDPO-style per-reward normalisation. They identify **active-group density** — the fraction of rollout groups where a reward produces nonzero relative advantage — as a hidden source of signal imbalance between rewards. DARA (Density-Aware Reward Aggregation) is a per-batch correction that upweights sparser rewards so each contributes proportionally to the policy update.

## How it works

Define **advantage energy** for a reward as the sum of its squared advantages over a batch. Under idealised GDPO normalisation, the authors show this energy equals the reward's active-group density $\rho_i$ (the fraction of rollout groups where reward $i$ gives a nonzero relative advantage). So sparser rewards contribute less gradient energy per batch, even though each one may be critical.

**The correction.** Scale each reward's advantage by $1/\sqrt{\rho_i}$, where $\rho_i$ is estimated per-batch from the fraction of rollout groups in which reward $i$ was active. The policy objective, group-relative normalisation, and reward composition are otherwise unchanged — DARA is a weight on advantages, not a new loss.

**Adaptive by construction.** As training progresses and reward activity drifts (format compliance saturates and becomes rarely active, length compliance becomes commonly active, etc.), $\rho_i$ changes each batch and the correction updates automatically — no schedule tuning.

## Why it matters

- On tool calling, DARA reaches high format compliance in **up to 26% fewer** training steps than GDPO.
- On math reasoning, DARA reaches near-saturated length compliance in **up to 65% fewer** steps.
- Final-performance competitive with GDPO on both tasks — DARA mostly accelerates learning rather than lifting the ceiling.
- Lands cleanly alongside [CorrGRPO](./corr-grpo.md): the two attack orthogonal axes of multi-reward imbalance (sparsity vs correlation/scale) and both plug into existing GRPO trainers.

## Gotchas & tricks

- The inverse-sqrt correction can blow up for a reward with near-zero active-group density; standard implementations add a small floor $\rho_i \leftarrow \max(\rho_i, \varepsilon)$.
- Batch size needs to be large enough that $\rho_i$ is a stable estimate — tiny batches give noisy density, which propagates to noisy advantage weighting.
- The theoretical equivalence is under *idealised* GDPO normalisation; in practice GDPO uses the empirical within-group std, so the energy = density equivalence is approximate. DARA still helps empirically.
- Only addresses the sparsity axis of multi-reward imbalance; combine with [CorrGRPO](./corr-grpo.md) when rewards also differ in scale/correlation.

## Sources

- Paper: *Make Sparse Rewards Count: Density-Aware Reward Aggregation for Multi-Reward RL* — Zheng et al., 2026 — [arXiv:2610.00574](https://arxiv.org/abs/2610.00574).
- Context: *GDPO* — the per-reward-normalised GRPO variant DARA augments.
- Context: *GRPO* — Shao et al., 2024 — the base RL algorithm.
