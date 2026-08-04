# CriPO — Criterion-Distilled Policy Optimization
*Depth — rubric-based RL with on-policy self-distillation for unexplored and suppressed criteria.*

**TL;DR:** Rubric-based RL has two blind spots. **Unexplored criteria** are those no rollout ever satisfies (no gradient signal). **Suppressed criteria** are satisfied by *some* rollouts but their token-level signal gets averaged out when the rubric is aggregated into a scalar reward. CriPO fixes both with model-internal self-teachers: a criterion-injection self-teacher writes reference outputs that a forward-KL loss injects into the base policy, and a counterfactual self-teacher locates the tokens in negative-advantage rollouts that satisfied a criterion and flips their token-level advantage positive.

**Prereqs:** [grpo.md](./grpo.md), [_rewards.md](./_rewards.md)
**Related:** [rlsvr.md](./rlsvr.md), [cscr.md](./cscr.md)

---

## What it is

An extension to rubric-based RL that treats missed-criterion signals as a first-class training loss, not just an aggregation artifact. Motivated by a stark measurement: **>57% of samples exhibit suppressed criteria, with 1.8 suppressed criteria per sample on average** during training.

Two teachers, both derived from the current policy — no external model, no train-inference prompt mismatch:

## How it works

**For unexplored criteria (nobody satisfies them):**
1. Add the criterion text to the prompt as a temporary hint → the *criterion-injection self-teacher* produces outputs that satisfy it.
2. Compute a *localized* forward-KL loss between the base policy (without the hint) and the injected teacher, restricted to token positions where the criterion is most likely realized.
3. This injects the missing behavior into the base policy without ever exposing the hint at inference time.

**For suppressed criteria (some rollouts satisfy them, but aggregate advantage is ≤ 0):**
1. Run a *counterfactual self-teacher* on the negative-advantage rollout: identify which spans of tokens correspond to the satisfied criterion (typically via a rubric-item classifier prompted at the token level).
2. Flip the token-level advantages on those spans from negative to positive, keeping the rest of the rollout's advantages unchanged.
3. Optimize with the modified per-token advantages.

## Why it matters

- The gradient signal in rubric-based RL is much richer than aggregation reveals; CriPO recovers most of it.
- ~2× fewer optimization steps to reach the same final performance on medical and scientific benchmarks vs. rubric-based RL baselines.
- Generalizes to any multi-criterion reward setup — the suppressed-criteria pathology is not rubric-specific.

## Gotchas & tricks

- Choice of criterion-item classifier matters: too-loose spans include noise; too-tight spans miss valuable tokens.
- The localized KL must be *localized*; a global forward-KL against the injected teacher over-anchors the policy to the injection distribution.
- Not a substitute for a good rubric — CriPO amplifies the criteria you specify; it does not discover new ones.

## Sources

- Paper: *Enhancing Rubric-based RL via Self-Distillation* — Xia et al., 2026 — [arXiv:2607.18082](https://arxiv.org/abs/2607.18082)
