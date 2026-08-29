# TTPO — Test-Time Policy Optimization
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A **label-free test-time training** procedure for reasoning models. Naïvely swapping majority-vote pseudo-labels into on-policy self-distillation (OPSD) or GRPO is fragile — one wrong vote corrupts the teacher for every rollout. TTPO exploits the empirical asymmetry that **rollouts disagreeing with the vote are almost always wrong regardless of whether the vote is right**, and uses that to build an asymmetric objective: OPSD-distill the agreeing rollouts, GRPO-penalize the disagreeing ones, with token-level down-weighting of already-converged positions and only-confident-error penalties. Matches label-supervised OPSD across five competition benchmarks with no labels; raises Qwen3-1.7B AIME/etc. from 38.0% → 45.2%.

**Prereqs:** [../grpo.md](../grpo.md), [../rlvr.md](../rlvr.md)
**Related:** [long-cot-rl.md](long-cot-rl.md), [../_rl.md](../_rl.md), [../_rewards.md](../_rewards.md)

---

## What it is

Test-time training (TTT) means updating the model on unlabeled test inputs before answering, to specialize to the current distribution. TTT for reasoning has been mostly majority-vote inference (many samples → vote), which improves accuracy but doesn't update weights. TTPO actually *trains* at test time, using the majority vote as a pseudo-label and structuring the objective so that pseudo-label noise doesn't blow up the update.

## How it works

For each unlabeled test prompt:

1. **Sample G rollouts** from the current policy (GRPO-style).
2. **Majority-vote pseudo-label** across the rollouts.
3. **Split by agreement** with the pseudo-label:
   - *Agreeing branch* → treated as demonstrations for **on-policy self-distillation (OPSD)** — the model is pulled toward these tokens.
   - *Disagreeing branch* → treated as negative rollouts for **grouped RL (GRPO-style)** — the model is pushed away from these tokens.
4. **Token-level refinement.**
   - On the OPSD branch, down-weight positions where the model is already highly confident (the pull adds nothing).
   - On the RL branch, penalize *only confident errors* — low-confidence disagreeing tokens are noise, not systematic mistakes to punish.
5. **Update, resample, repeat** for a few steps per test prompt (or per batch of related test prompts).

The asymmetry is the key: even when the majority vote is wrong, the *disagreeing* rollouts are still mostly wrong (they were minority for a reason). So penalizing them is safe. The agreeing branch is where vote correctness matters, and OPSD's KL-anchored update is conservative enough to survive occasional bad votes.

## Why it matters

- **Label-free TTT for reasoning that actually trains.** Prior TTT was inference-time voting; TTPO updates weights and matches label-supervised OPSD.
- **Cross-task generalization is retained.** Unlike some TTT methods that overfit the target distribution, the paper reports the base capability is preserved.
- **Practical for per-user or per-domain specialization.** Users don't have labels; TTPO turns their queries into weight updates.

## Gotchas & tricks

- **Needs G large enough for a real vote.** G=8 minimum in the paper; G=16+ recommended.
- **Base model must clear a competence floor.** If the base can't produce any correct rollouts, the vote is uniformly wrong and both branches degrade.
- **KL anchor to the base model is still on** — TTPO is not standalone RL, it inherits GRPO's reference-model KL.
- **Token-level gates are the ML judgment.** The confidence thresholds for the OPSD down-weight and the RL "confident errors" mask are hyperparameters the paper tunes per benchmark family.
- **Compute cost is nontrivial** — G rollouts + gradient step per test prompt is much more expensive than single-shot inference. Batch related prompts to amortize.

## Sources

- Paper: *TTPO: Test-Time Policy Optimization* — Wang et al. (Zhejiang / Tencent), 2026 — arXiv:2608.27448.
