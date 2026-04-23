# Online Policy Mirror Descent (Kimi k1.5's RL Objective)
*Depth — the RL objective used by Kimi k1.5: a KL-regularized expected-reward problem solved iteratively via an ℓ₂-regression surrogate on log policy ratios.*

**TL;DR:** For each iteration, fix the current policy `π_{θ_i}` as the **reference**. Solve the KL-regularized problem `max E[r] − τ · KL(π_θ || π_{θ_i})` via a closed-form identity that expresses the reward as `τ · log(π* / π_{θ_i})`. The surrogate loss is a **squared-error regression** of `τ · log(π_θ / π_{θ_i})` onto the centered reward `r − r̄`, estimated on `k` rollouts from the reference. No clip, no importance-ratio trick, no value network. The ℓ₂-regularization on log-ratios plays the role of the KL penalty. Mechanically cousins with GRPO (both value-free, both group-relative-reward-based); the difference is surrogate form (ℓ₂ vs clipped ratio) and reference-update schedule (per-iteration vs rolling).

**Prereqs:** [_rl](../_rl.md), [ppo](../ppo.md), [grpo](../grpo.md)
**Related:** [long-cot-rl](long-cot-rl.md) · [length-penalty](length-penalty.md) · [long2short](long2short.md) · [kimi-k1-5 case study](../../case-studies/kimi-k1-5.md)

---

## What it is

The policy-optimization algorithm introduced by Kimi k1.5 (Moonshot AI, 2025, arXiv 2501.12599) for long-CoT RL. Framed in the paper as "online policy mirror descent" — each RL iteration is one step of mirror descent with the **current policy as the reference**, solved via a regression surrogate rather than a clipped policy gradient.

Three defining properties:

1. **Reference = current policy.** At iteration `i`, the reference policy `π_{θ_i}` is the policy at the *start* of this iteration. After the iteration's gradient updates finish, `θ_{i+1}` becomes the new reference for iteration `i+1`.
2. **ℓ₂ regression surrogate.** The gradient has two terms: a REINFORCE-style policy gradient with a mean-reward baseline, plus a squared-loss regularizer on `log(π_θ / π_{θ_i})` that plays the role of the trust-region constraint.
3. **No value network, no clip, no PRM, no MCTS.** Outcome reward only; credit assignment happens implicitly via the KL regularization and the reward baseline. This is a deliberate design choice explained in the paper (Sec. 2.3.2) — in long-CoT RL, tokens on a wrong branch that later recovers should be reinforced, and a value network would perversely penalize them.

The algorithm is structurally similar to [GRPO](../grpo.md) — both are value-network-free, both use a group-relative mean as the baseline — but the derivation path is different: GRPO is PPO-with-group-baseline; mirror descent is KL-regularized RL with a closed-form optimum plus a regression surrogate.

---

## How it works

### Prereq: mirror descent in one sentence

Mirror descent solves `max f(θ) − τ · D(θ || θ_old)` where `D` is a divergence. The update rule is implicit: each step, you set `θ_old` to the current parameters and take a step of the regularized problem. For RL with KL regularization, this gives the classical result that the optimal policy under `τ · KL` has a Gibbs-Boltzmann form relative to the reference — which is exactly what [DPO](../dpo.md) exploits for preference learning.

Kimi k1.5 applies the same identity but solves it **iteratively and online** with new rollouts each iteration, hence "online policy mirror descent."

### Variables

From Sec. 2.3.1:
- `x` = problem (prompt), `y*` = ground-truth answer.
- `z = (z_1, …, z_m)` = chain-of-thought tokens (each `z_i` can itself be a sub-sequence).
- `y` = final answer. Both `z` and `y` sampled auto-regressively from `π_θ(·|x)`.
- `r(x, y, y*) ∈ {0, 1}` = binary outcome reward (rule-based for verifiable problems, CoT-RM for free-form).
- `π_{θ_i}` = reference policy at iteration i.
- `τ > 0` = KL regularization temperature.

### Base objective (Eq. 1)

```
max_θ  E_{(x, y*) ~ D, (y, z) ~ π_θ} [ r(x, y, y*) ]
```

Pure expected reward — no regularization.

### KL-regularized iterative objective (Eq. 2)

Each mirror-descent iteration optimizes:

```
max_θ  E_{(x, y*) ~ D} [
    E_{(y, z) ~ π_θ} [ r(x, y, y*) ]  −  τ · KL( π_θ(·|x) || π_{θ_i}(·|x) )
]
```

The KL direction is `KL(π_θ || π_{θ_i})` — forward-KL of the learner against the reference. Same direction as PPO's trust-region penalty. The paper writes it as an *exact distributional KL* in the derivation, not a Monte-Carlo estimator.

### Closed-form optimum of the KL-regularized problem

Gibbs-Boltzmann form (follows from Gibbs' inequality, same derivation as DPO uses):

```
π*(y, z | x)  =  π_{θ_i}(y, z | x) · exp( r(x, y, y*) / τ )  /  Z(x)

Z(x)  =  Σ_{y', z'}  π_{θ_i}(y', z' | x) · exp( r(x, y', y*) / τ )
```

`Z(x)` is the partition function — sum over all `(y', z')` continuations of `x` under the reference, weighted by `exp(reward/τ)`.

### Optimality identity

Take logs of the per-sample optimality condition:

```
r(x, y, y*)  −  τ · log Z(x)  =  τ · log ( π*(y, z | x) / π_{θ_i}(y, z | x) )
```

Read it: at optimum, `τ · log(π_θ / π_{θ_i})` should equal the **centered reward** `r − τ log Z`. This is the regression target.

### The surrogate loss

The loss that `θ` is trained to minimize:

```
L(θ)  =  E_{(x, y*) ~ D} [
    E_{(y, z) ~ π_{θ_i}} [
        (  r(x, y, y*)  −  τ · log Z(x)
           −  τ · log( π_θ(y, z | x) / π_{θ_i}(y, z | x) )  )^2
    ]
]
```

This is the most important formula. Read it as: **"make `τ · log(π_θ / π_{θ_i})` a regressor for the centered reward, trained on rollouts drawn from the reference policy `π_{θ_i}`."**

Properties:
- **Squared-error loss**, not a clipped policy gradient.
- **Rollouts from the reference** `π_{θ_i}`, not from the current `π_θ` — off-policy within an iteration.
- **No importance-ratio clipping**. The ℓ₂ term regularizes the learner toward the reference, which is the trust-region enforcement.

### Estimating the baseline `τ · log Z(x)`

In principle, Monte-Carlo:

```
τ · log Z(x)  ≈  τ · log ( (1/k) · Σ_{j=1..k}  exp( r(x, y_j, y*) / τ ) )
```

with `k` reference rollouts. In practice, the paper reports that using the **empirical mean reward** instead "yields effective practical results" (Sec. 2.3.2):

```
r̄(x)  =  mean( r(x, y_1, y*), …, r(x, y_k, y*) )
```

Justification: as `τ → ∞`, `τ · log Z(x) → E_{π_{θ_i}}[r]`, so the mean is the right limit. This is the same baseline shape [GRPO](../grpo.md) uses — minus GRPO's standard-deviation normalization (z-score). Mirror descent doesn't z-score.

### The gradient (Eq. 3)

With `k` rollouts `(y_j, z_j) ~ π_{θ_i}` per problem:

```
g  =  (1/k) · Σ_{j=1..k} [
        ∇_θ log π_θ(y_j, z_j | x)  ·  ( r(x, y_j, y*)  −  r̄ )
        −  (τ / 2) · ∇_θ ( log( π_θ(y_j, z_j | x) / π_{θ_i}(y_j, z_j | x) ) )^2
      ]
```

Two components:

- **First term**: standard REINFORCE / policy gradient with the mean-reward baseline. Identical in spirit to GRPO's group-relative advantage (without GRPO's `/std` normalization).
- **Second term**: the **ℓ₂-regularization on log-ratios**, pulling `π_θ` back toward `π_{θ_i}`. This is the trust-region enforcement and it replaces PPO's clip.

The paper's own phrasing (Sec. 2.3.2): *"this gradient resembles the policy gradient of (2) using the mean of sampled rewards as the baseline. The main differences are that the responses are sampled from π_{θ_i} rather than on-policy, and an ℓ₂-regularization is applied."*

### Per-iteration mechanics

```
Initialize θ_0.
For iteration i = 0, 1, 2, …:
    Reference ← π_{θ_i} (frozen for this iteration)
    For problem x in batch:
        Sample k rollouts (y_j, z_j) ~ π_{θ_i}
        Compute reward r_j for each; compute r̄ = mean(r_j)
    For several gradient steps:
        Update θ via the gradient in Eq. 3 (evaluated at current θ)
    θ_{i+1} ← current θ
    Reset optimizer state              ← Sec. 2.3.2, explicit design choice
```

Why the **optimizer reset**: each iteration has a new reference policy, so the optimization landscape has shifted. Adam's accumulated momentum from the previous iteration is now in the wrong direction. Resetting guarantees clean convergence per iteration. Unusual but principled.

### Contrast with GRPO

The two algorithms are cousins, not identical. Neither contrast is stated in the Kimi paper (which never names GRPO); this comparison is constructed from the two papers together.

| Aspect | [GRPO](../grpo.md) (DeepSeekMath) | k1.5 mirror descent |
|---|---|---|
| Surrogate form | PPO-style **clipped ratio** (`min(r·A, clip(r)·A)`) | **ℓ₂ regression** on log-ratios |
| Baseline | Group mean **with std normalization** (z-score) | Group mean (**no std normalization**) |
| KL regularization | Explicit `β · KL(π_θ || π_ref)` added to loss | Implicit via ℓ₂ on `log(π_θ / π_{θ_i})` |
| Off-policy handling | Importance ratio + clip | Ratio appears inside ℓ₂ regularizer only |
| Reference update | Rolling (updated every few steps within an RL round) | Per-iteration (frozen for the whole iteration) |
| Optimizer | Persistent across updates | **Reset at each iteration** |
| Advantage granularity | Same per-token (group-relative advantage broadcast over tokens) | Same per-response (reward is per-response) |

Mechanically, both algorithms are value-network-free and use a group-relative mean baseline. The derivation path, reference schedule, and trust-region implementation differ. Whether one beats the other at equal compute is not empirically settled — no paper has run them head-to-head with controlled hyperparameters.

---

## Why it matters

- **Principled trust region.** The ℓ₂ penalty on log-ratios comes out of the mirror-descent derivation — it's not a PPO heuristic. If you care about the connection between the loss and the mathematical problem being solved, mirror descent is cleaner than PPO.
- **Simpler than PPO-style clipping.** No clip threshold, no asymmetric `min(clip, ratio)` logic. One regression loss. Arguably easier to implement and debug than a proper PPO clip, though the clip has its own advantages (bounded updates by construction).
- **Absorbs the off-policy issue into the regularizer.** Rollouts are from the reference (off-policy w.r.t. current θ). PPO handles this with importance ratios and clipping; mirror descent handles it by assuming the policies are close (the ℓ₂ term enforces this) and accepting bias in the baseline. Works when the per-iteration update is small.
- **Opens the door to DPO-mirror-descent unification.** DPO and this algorithm share the same mirror-descent derivation structure (Gibbs optimum, log-ratio identity). DPO is the offline-preference-data special case; online mirror descent is the online-reward special case. Understanding one helps with the other.
- **Canonical reference for frontier reasoning RL.** Kimi k1.5 is the only widely-cited contemporary alternative to GRPO for long-CoT RL at scale. If the field branches, this is one of the two trunks.

---

## Gotchas & tricks

- **The "Z(x) → empirical mean" approximation breaks for small `τ`.** The identity `τ log Z → E[r]` only holds as `τ → ∞`. For small `τ` (strong regularization), the mean-reward baseline biases the centered reward. The paper uses the empirical mean despite this, because it works — but the approximation is load-bearing. If you tune `τ` small, revisit.
- **Optimizer reset matters.** Carrying Adam state across iterations conflates two different optimization problems (different references, different solutions). Don't skip this.
- **`k` rollouts per problem is undisclosed.** The paper uses `k` symbolically throughout. Open implementations default to `k = 8-16`.
- **ℓ₂ regularizer weight `τ/2` is the KL strength.** Large `τ` → strong regularization → small per-iteration updates. Small `τ` → weak regularization → large updates but potentially unstable (reference-iterate stale). Tune like PPO's β.
- **Rollouts are fully off-policy within an iteration.** Unlike GRPO (which re-samples between inner mini-batch updates), mirror descent uses the same `k` rollouts from `π_{θ_i}` for every gradient step in an iteration. The ℓ₂ regularizer is supposed to keep this safe, but many inner steps per iteration can push `π_θ` far from the rollouts and invalidate the regression target. Keep inner steps per iteration small.
- **No clipping means no hard upper bound on per-step update size.** The regularizer is soft; a bad mini-batch can push `π_θ` far. Monitor `log(π_θ / π_{θ_i})` magnitudes during training.
- **Length penalty interacts.** k1.5 adds a length-penalty reward term (see [length-penalty](length-penalty.md)) *on top* of the outcome reward before computing `r̄` and `r − r̄`. The ℓ₂ regression still applies to the combined reward.
- **No explicit per-token advantage.** Like GRPO, mirror descent assigns the same per-response reward to every token. For verifiable outcome rewards this is fine; for token-level reward signals (PRM-style), mirror descent would need adaptation.
- **Explicit contrast with R1 is not in the paper.** The paper doesn't name R1 / GRPO / DeepSeekMath anywhere in the text (Jan 2025 simultaneity). Any claims about "Kimi k1.5 is better/worse than R1 at equal compute" are external interpretations.

---

## Sources

- Paper: *Kimi k1.5: Scaling Reinforcement Learning with LLMs* — Moonshot AI (Kimi Team), 2025, arXiv 2501.12599 — introduces the algorithm; Sec. 2.3.1 (setup), Sec. 2.3.2 (derivation + gradient Eq. 3).
- Paper: *Direct Preference Optimization* — Rafailov et al., 2023, arXiv 2305.18290 — same closed-form optimum, used for preferences instead of online rewards.
- Paper: *DeepSeekMath* — Shao et al., 2024, arXiv 2402.03300 — introduces [GRPO](../grpo.md), the value-network-free PPO cousin this algorithm sits alongside.
- Paper: *Proximal Policy Optimization* — Schulman et al., 2017, arXiv 1707.06347 — [PPO](../ppo.md), the baseline this algorithm replaces.
- Textbook: *Convex Optimization* — Boyd & Vandenberghe, 2004 — mirror descent (Section 11.4).
