# Rewardable Support (Verifier-Induced Support Reshaping)
*Depth — a diagnostic showing that RLVR on task A can collapse the set of trajectories a later verifier on task B can find.*

**TL;DR:** Under a fixed rollout budget, RLVR mostly *reranks* the base policy's opening tokens rather than expanding what it can produce. Optimizing that ranking against verifier A concentrates probability on A-preferred openings, causally shrinking the trajectories that verifier B can still reach — the *effective rewardable support* for B. On Qwen3-8B-Base, Math-RLVR raises IFEval pass@1 by 6.5 pp but drops best@32 by 9.8 pp; IF-RLVR does the mirror-image damage to math. Reference-KL, routing priors, and on-policy distillation only partially preserve cross-task support.

**Prereqs:** [rlvr](rlvr.md), [grpo](grpo.md)
**Related:** [rejection-sampling](rejection-sampling.md), [../evaluation/ifeval.md](../evaluation/ifeval.md)

---

## What it is

A metric-plus-diagnostic for multi-objective post-training. **Effective rewardable support** on task T is the set of prompts for which the policy produces at least one T-verifier-passing trajectory within a fixed rollout budget `k` (i.e., prompts where `best@k > 0`). Verifier-induced support reshaping is the phenomenon that RLVR on a different task T' shrinks the rewardable support of T — silently, in a way pass@1 improvements on T' hide.

## How it works

Diagnostic protocol:

1. **Measure baseline support.** For every task T you care about, compute `S_T(π_0) = { prompts : best@k(π_0) > 0 }` on the base policy `π_0`.
2. **Train.** Run RLVR on task T'.
3. **Re-measure.** Compute `S_T(π_1)` on the trained policy for each T ≠ T'.
4. **Report both.** pass@1 on T' captures gain; `|S_T(π_1)| / |S_T(π_0)|` captures cross-task support cost.

Mechanism: token-distribution analyses show the RLVR-induced shift concentrates in the first few response tokens. Controlled opening-token interventions confirm that the selected opening *causally* affects downstream searchability — the base policy already contained both opening styles, RLVR just rebalanced them.

## Why it matters

- **Multi-stage RLVR blind spot.** Frontier post-training runs stack RLVR stages (math → IF → tool-use → …). Each stage looks fine on its own leaderboard while quietly narrowing the policy's basin for later stages.
- **Best@k must be reported.** pass@1 alone hides the collapse. This paper makes a scientific case for tracking best@k across held-out tasks throughout multi-objective RL.
- **Mitigations are only partial.** The commonly cited fixes — KL to a reference policy, routing priors, on-policy distillation — reduce but do not eliminate the effect. Solving cross-task support preservation is an open problem.

## Gotchas & tricks

- Support shrinkage is only visible at meaningful `k` (32+); a `k=8` sweep can miss it entirely.
- Directionality matters: Math-RLVR shifts IF openings toward step-by-step preambles; IF-RLVR shifts math openings toward direct answers. Each pattern is legible in the first ~5 tokens.
- Sequential training with the *opposite* verifier partially recovers but does not restore original support — the reshaping is not a symmetric operation.

## Sources

- Verifier-Induced Support Reshaping in On-Policy Optimization — Shaohang Wei et al., 2026 — [arXiv:2608.00220](https://arxiv.org/abs/2608.00220)
