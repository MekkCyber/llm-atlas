# Reward Discretization
*Depth — a training-free fix for reward-model oversensitivity that reduces reward hacking without losing discrimination.*

**TL;DR:** Continuous reward-model scores look fine-grained — they assign *different* scores to *equally good* responses. That's not informative; it's **oversensitivity**, and policies trained against it overfit to spurious score variation (reward hacking). Reward discretization replaces the continuous head with a Monte-Carlo-dropout-based discrete cluster assignment at inference time. Theoretically: there exist discretizations that preserve the RM's ability to separate good from bad while collapsing the spurious continuous structure. Empirically: less reward hacking, better policies, no retraining needed.

**Prereqs:** [_rewards.md](_rewards.md), [grpo.md](grpo.md), [ppo.md](ppo.md)
**Related:** [cot-reward-model.md](cot-reward-model.md), [reasoning/orm.md](reasoning/orm.md), [rlvr.md](rlvr.md)

---

## What it is

A drop-in transformation applied to any neural reward model at inference time. Replaces "RM accuracy" — which conflates two things — with two separate metrics:

- **Discriminative ability.** Can the RM tell a good response from a bad one? (What the field has been measuring.)
- **Specificity.** Does the RM assign *the same* score to *equally good* responses, or does it spuriously differentiate them?

A perfectly accurate RM can still be terrible at specificity. Discretization is the inference-time procedure that recovers specificity at minimal cost to discrimination.

## How it works

The recipe is small:

1. Take any trained neural RM.
2. At inference, run $N$ forward passes with **Monte Carlo dropout** enabled.
3. Cluster the resulting score distribution into $K$ discrete bins (paper presents a procedure to pick $K$ automatically).
4. Use the cluster centroid as the reward for RL training.

That's it. No retraining, no architectural change, $\sim N\times$ inference cost during reward computation.

Theoretical guarantee: there always exists a discretization that reduces oversensitivity strictly more than it reduces discriminative ability — the two are decoupled, so you can trade off against one without hurting the other.

## Why it matters

- Cheap to deploy. Any team already running PPO/GRPO with a learned RM can add it in a day.
- Addresses a failure mode the field had no name for. The intuition was always "more sensitive RM = better RL signal"; the paper shows that's wrong above a threshold.
- Sharpens the measurement vocabulary for RMs going forward: report discrimination *and* specificity, not just accuracy.

## Gotchas & tricks

- MC-dropout only works on RMs that have dropout layers (most do). For RMs without dropout, ensembling over training-seed checkpoints is the analogue.
- Discretization granularity ($K$) matters: too coarse and you lose discrimination; too fine and you're back to oversensitivity. The paper's automatic procedure is the safe default.
- Pairs well with [cot-reward-model](cot-reward-model.md): CoT RMs reduce oversensitivity *during training* by giving the verifier explicit reasoning room; discretization reduces it *post-hoc* by collapsing spurious score variation. Either path works; both at once is overkill.
- Doesn't help rule-based verifiers (RLVR) — they're already discrete by construction.

## Sources

- Paper: *Discretizing Reward Models* — Viswanathan, Wang, Hazarika, Nagpal, Wu, Neubig, Mao, 2026 — [arXiv:2606.21795](https://arxiv.org/abs/2606.21795). CMU / Meta Superintelligence Labs.
