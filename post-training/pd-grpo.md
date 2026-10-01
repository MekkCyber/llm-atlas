# PD-GRPO — Pairwise Dual-Group Relative Policy Optimization
*Depth — a GRPO variant that uses pairwise supervision while keeping fine-grained pointwise scoring, built for visual reward-model training.*

**TL;DR:** Introduced in the **Thinking Reward Model** (TRM) paper (Bai et al., 2026) for visual generation, **PD-GRPO** adapts [GRPO](./grpo.md) so that pairwise preference supervision drives learning *without* collapsing pointwise scores into polarized extremes. Uses two groups — one for the preferred candidate and one for the dispreferred — and computes group-relative advantages inside each, so the reward model keeps fine-grained *pointwise* scores while learning *pairwise* discrimination.

**Prereqs:** [grpo.md](./grpo.md), [ppo.md](./ppo.md), [_rewards.md](./_rewards.md)
**Related:** [reasoning/prm.md](./reasoning/prm.md), [dpo.md](./dpo.md), [cot-reward-model.md](./cot-reward-model.md)

---

## What it is

Standard pairwise preference optimization (DPO-style; see [dpo.md](./dpo.md)) collapses to a binary objective — the preferred wins, the dispreferred loses. For a reward model that must *also* output pointwise scores (needed for best-of-N, PPO shaping, rubric-grounded scoring), this binary signal induces **score polarization**: all preferred candidates saturate near the top and all dispreferred near the bottom. Fine-grained ordering between candidates of the same side collapses.

PD-GRPO's fix: separate the pairwise constraint from the pointwise distribution. Each pair contributes two groups ($G_+ = \{\text{preferred candidates}\}$ and $G_- = \{\text{dispreferred candidates}\}$), and GRPO's group-mean advantages are computed *inside each group*. The pairwise supervision only sets the *relative* positioning of the two group means; it no longer dictates the shape within.

## How it works

For each pairwise example $(x, y_+, y_-)$:

1. **Group formation.** Form $G_+$ from candidates intended to score high (e.g. $y_+$ and nearby high-reward samples) and $G_-$ from candidates intended to score low. Both groups may contain the same model's samples ranked by TRM's rubric step.
2. **Within-group advantages.** For each group, compute GRPO-style advantages (mean and std over group members' rewards). This gives a *shape-preserving* signal: within $G_+$, better members still get positive advantages; within $G_-$, worse members still get negative ones.
3. **Pairwise constraint.** Add a cross-group term requiring $\bar r_+ > \bar r_-$ by a margin. This is the pairwise supervision, but it only pulls the group *means*, not individual scores.
4. **Combine.** The PD-GRPO loss is the sum of the two within-group GRPO terms plus the cross-group margin term, with the KL-to-reference penalty inherited from GRPO.

## Why it matters

- **Fine-grained pointwise + pairwise in one loss.** TRM uses PD-GRPO to train a visual reward model that gives rubric-grounded pointwise scores *and* respects pairwise preferences. Prior work had to pick one.
- **Portable beyond visual rewards.** Any reward model trained from a mix of pairwise comparisons (cheap to collect) and pointwise scores (expensive but informative) can use PD-GRPO instead of multi-head loss hacks.
- **Reads naturally against DPO.** Where [DPO](./dpo.md) replaces RL with a direct pairwise classification objective and gives up pointwise granularity, PD-GRPO keeps the RL structure and uses dual groups to preserve granularity.

## Gotchas & tricks

- **Group construction matters.** If both groups are near-identical in reward, the pairwise margin term stops pulling and only within-group signal remains. Mine groups so the preferred side is noticeably higher.
- **Needs more rollouts per pair.** GRPO's cost is $O(G)$ per prompt; PD-GRPO is $O(G_+ + G_-)$. The within-group granularity costs a factor of ~2 over vanilla GRPO.
- **KL to reference is still essential.** Dropping the KL term causes reward hacking the same way vanilla GRPO does.
- **Normalization edge case.** If one group collapses to a single sample, within-group std is zero. Fall back to raw-advantage computation as in standard GRPO.

## Sources

- Paper: *Think Before You Score: Thinking Reward Model for Visual Generation* — Bai, Tang, Shi, Wang, Liu, Su, Zhu, Wang, Diao, Wang, Gu, Zhang, 2026 — [arXiv:2609.37372](https://arxiv.org/abs/2609.37372) — introduces PD-GRPO as the reward-model trainer behind TRM.
