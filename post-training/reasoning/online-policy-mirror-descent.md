# Online Policy Mirror Descent (Kimi k1.5's RL Objective)
*Depth — the RL objective used by Kimi k1.5: a KL-regularized expected-reward problem solved iteratively via an $\ell_2$-regression surrogate on log policy ratios.*

**TL;DR:** For each iteration, fix the current policy $\pi_{\theta_i}$ as the **reference**. Solve the KL-regularized problem $\max\,\mathbb{E}[r] - \tau \cdot \mathrm{KL}(\pi_\theta \,\|\, \pi_{\theta_i})$ via a closed-form identity that expresses the reward as $\tau \cdot \log(\pi^* / \pi_{\theta_i})$. The surrogate loss is a **squared-error regression** of $\tau \cdot \log(\pi_\theta / \pi_{\theta_i})$ onto the centered reward $r - \bar{r}$, estimated on $k$ rollouts from the reference. No clip, no importance-ratio trick, no value network. The $\ell_2$-regularization on log-ratios plays the role of the KL penalty. Mechanically cousins with GRPO (both value-free, both group-relative-reward-based); the difference is surrogate form ($\ell_2$ vs clipped ratio) and reference-update schedule (per-iteration vs rolling).

**Prereqs:** [_rl](../_rl.md), [ppo](../ppo.md), [grpo](../grpo.md)
**Related:** [long-cot-rl](long-cot-rl.md) · [length-penalty](length-penalty.md) · [long2short](long2short.md) · [kimi-k1-5 case study](../../case-studies/kimi-k1-5.md)

---

## What it is

The policy-optimization algorithm introduced by Kimi k1.5 (Moonshot AI, 2025, arXiv 2501.12599) for long-CoT RL. Framed in the paper as "online policy mirror descent" — each RL iteration is one step of mirror descent with the **current policy as the reference**, solved via a regression surrogate rather than a clipped policy gradient.

Three defining properties:

1. **Reference = current policy.** At iteration $i$, the reference policy $\pi_{\theta_i}$ is the policy at the *start* of this iteration. After the iteration's gradient updates finish, $\theta_{i+1}$ becomes the new reference for iteration $i+1$.
2. **$\ell_2$ regression surrogate.** The gradient has two terms: a REINFORCE-style policy gradient with a mean-reward baseline, plus a squared-loss regularizer on $\log(\pi_\theta / \pi_{\theta_i})$ that plays the role of the trust-region constraint.
3. **No value network, no clip, no PRM, no MCTS.** Outcome reward only; credit assignment happens implicitly via the KL regularization and the reward baseline. This is a deliberate design choice explained in the paper (Sec. 2.3.2) — in long-CoT RL, tokens on a wrong branch that later recovers should be reinforced, and a value network would perversely penalize them.

The algorithm is structurally similar to [GRPO](../grpo.md) — both are value-network-free, both use a group-relative mean as the baseline — but the derivation path is different: GRPO is PPO-with-group-baseline; mirror descent is KL-regularized RL with a closed-form optimum plus a regression surrogate.

---

## How it works

### Prereq: mirror descent in one sentence

Mirror descent solves $\max\, f(\theta) - \tau \cdot D(\theta \,\|\, \theta_{\text{old}})$ where $D$ is a divergence. The update rule is implicit: each step, you set $\theta_{\text{old}}$ to the current parameters and take a step of the regularized problem. For RL with KL regularization, this gives the classical result that the optimal policy under $\tau \cdot \mathrm{KL}$ has a Gibbs-Boltzmann form relative to the reference — which is exactly what [DPO](../dpo.md) exploits for preference learning.

Kimi k1.5 applies the same identity but solves it **iteratively and online** with new rollouts each iteration, hence "online policy mirror descent."

### Variables

From Sec. 2.3.1:
- $x$ = problem (prompt), $y^*$ = ground-truth answer.
- $z = (z_1, \ldots, z_m)$ = chain-of-thought tokens (each $z_i$ can itself be a sub-sequence).
- $y$ = final answer. Both $z$ and $y$ sampled auto-regressively from $\pi_\theta(\,\cdot\, \mid x)$.
- $r(x, y, y^*) \in \{0, 1\}$ = binary outcome reward (rule-based for verifiable problems, CoT-RM for free-form).
- $\pi_{\theta_i}$ = reference policy at iteration $i$.
- $\tau > 0$ = KL regularization temperature.

### Base objective (Eq. 1)

$$
\max_\theta\; \mathbb{E}_{(x, y^*) \sim \mathcal{D},\, (y, z) \sim \pi_\theta} [\, r(x, y, y^*) \,]
$$

Pure expected reward — no regularization.

### KL-regularized iterative objective (Eq. 2)

Each mirror-descent iteration optimizes:

$$
\max_\theta\; \mathbb{E}_{(x, y^*) \sim \mathcal{D}} \!\left[ \mathbb{E}_{(y, z) \sim \pi_\theta} [\, r(x, y, y^*) \,] - \tau \cdot \mathrm{KL}( \pi_\theta(\,\cdot\, \mid x) \,\|\, \pi_{\theta_i}(\,\cdot\, \mid x) ) \right]
$$

The KL direction is $\mathrm{KL}(\pi_\theta \,\|\, \pi_{\theta_i})$ — forward-KL of the learner against the reference. Same direction as PPO's trust-region penalty. The paper writes it as an *exact distributional KL* in the derivation, not a Monte-Carlo estimator.

### Closed-form optimum of the KL-regularized problem

Gibbs-Boltzmann form (follows from Gibbs' inequality, same derivation as DPO uses):

$$
\pi^*(y, z \mid x) = \frac{\pi_{\theta_i}(y, z \mid x) \cdot \exp( r(x, y, y^*) / \tau )}{Z(x)}
$$

$$
Z(x) = \sum_{y', z'} \pi_{\theta_i}(y', z' \mid x) \cdot \exp( r(x, y', y^*) / \tau )
$$

$Z(x)$ is the partition function — sum over all $(y', z')$ continuations of $x$ under the reference, weighted by $\exp(\mathrm{reward}/\tau)$.

### Optimality identity

Take logs of the per-sample optimality condition:

$$
r(x, y, y^*) - \tau \cdot \log Z(x) = \tau \cdot \log\!\left( \frac{\pi^*(y, z \mid x)}{\pi_{\theta_i}(y, z \mid x)} \right)
$$

Read it: at optimum, $\tau \cdot \log(\pi_\theta / \pi_{\theta_i})$ should equal the **centered reward** $r - \tau \log Z$. This is the regression target.

### The surrogate loss

The loss that $\theta$ is trained to minimize:

$$
L(\theta) = \mathbb{E}_{(x, y^*) \sim \mathcal{D}} \!\left[ \mathbb{E}_{(y, z) \sim \pi_{\theta_i}} \!\left[ \left( r(x, y, y^*) - \tau \cdot \log Z(x) - \tau \cdot \log\!\frac{\pi_\theta(y, z \mid x)}{\pi_{\theta_i}(y, z \mid x)} \right)^2 \right] \right]
$$

This is the most important formula. Read it as: **"make $\tau \cdot \log(\pi_\theta / \pi_{\theta_i})$ a regressor for the centered reward, trained on rollouts drawn from the reference policy $\pi_{\theta_i}$."**

Properties:
- **Squared-error loss**, not a clipped policy gradient.
- **Rollouts from the reference** $\pi_{\theta_i}$, not from the current $\pi_\theta$ — off-policy within an iteration.
- **No importance-ratio clipping**. The $\ell_2$ term regularizes the learner toward the reference, which is the trust-region enforcement.

### Estimating the baseline $\tau \cdot \log Z(x)$

In principle, Monte-Carlo:

$$
\tau \cdot \log Z(x) \approx \tau \cdot \log\!\left( \frac{1}{k} \sum_{j=1}^{k} \exp( r(x, y_j, y^*) / \tau ) \right)
$$

with $k$ reference rollouts. In practice, the paper reports that using the **empirical mean reward** instead "yields effective practical results" (Sec. 2.3.2):

$$
\bar{r}(x) = \mathrm{mean}( r(x, y_1, y^*), \ldots, r(x, y_k, y^*) )
$$

Justification: as $\tau \to \infty$, $\tau \cdot \log Z(x) \to \mathbb{E}_{\pi_{\theta_i}}[r]$, so the mean is the right limit. This is the same baseline shape [GRPO](../grpo.md) uses — minus GRPO's standard-deviation normalization (z-score). Mirror descent doesn't z-score.

### The gradient (Eq. 3)

With $k$ rollouts $(y_j, z_j) \sim \pi_{\theta_i}$ per problem:

$$
g = \frac{1}{k} \sum_{j=1}^{k} \left[ \nabla_\theta \log \pi_\theta(y_j, z_j \mid x) \cdot ( r(x, y_j, y^*) - \bar{r} ) - \frac{\tau}{2} \cdot \nabla_\theta \!\left( \log\!\frac{\pi_\theta(y_j, z_j \mid x)}{\pi_{\theta_i}(y_j, z_j \mid x)} \right)^2 \right]
$$

Two components:

- **First term**: standard REINFORCE / policy gradient with the mean-reward baseline. Identical in spirit to GRPO's group-relative advantage (without GRPO's `/std` normalization).
- **Second term**: the **$\ell_2$-regularization on log-ratios**, pulling $\pi_\theta$ back toward $\pi_{\theta_i}$. This is the trust-region enforcement and it replaces PPO's clip.

The paper's own phrasing (Sec. 2.3.2): *"this gradient resembles the policy gradient of (2) using the mean of sampled rewards as the baseline. The main differences are that the responses are sampled from $\pi_{\theta_i}$ rather than on-policy, and an $\ell_2$-regularization is applied."*

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
| Surrogate form | PPO-style **clipped ratio** ($\min(r \cdot A, \mathrm{clip}(r) \cdot A)$) | **$\ell_2$ regression** on log-ratios |
| Baseline | Group mean **with std normalization** (z-score) | Group mean (**no std normalization**) |
| KL regularization | Explicit $\beta \cdot \mathrm{KL}(\pi_\theta \,\|\, \pi_{\text{ref}})$ added to loss | Implicit via $\ell_2$ on $\log(\pi_\theta / \pi_{\theta_i})$ |
| Off-policy handling | Importance ratio + clip | Ratio appears inside $\ell_2$ regularizer only |
| Reference update | Rolling (updated every few steps within an RL round) | Per-iteration (frozen for the whole iteration) |
| Optimizer | Persistent across updates | **Reset at each iteration** |
| Advantage granularity | Same per-token (group-relative advantage broadcast over tokens) | Same per-response (reward is per-response) |

Mechanically, both algorithms are value-network-free and use a group-relative mean baseline. The derivation path, reference schedule, and trust-region implementation differ. Whether one beats the other at equal compute is not empirically settled — no paper has run them head-to-head with controlled hyperparameters.

---

## Why it matters

- **Principled trust region.** The $\ell_2$ penalty on log-ratios comes out of the mirror-descent derivation — it's not a PPO heuristic. If you care about the connection between the loss and the mathematical problem being solved, mirror descent is cleaner than PPO.
- **Simpler than PPO-style clipping.** No clip threshold, no asymmetric $\min(\mathrm{clip}, \mathrm{ratio})$ logic. One regression loss. Arguably easier to implement and debug than a proper PPO clip, though the clip has its own advantages (bounded updates by construction).
- **Absorbs the off-policy issue into the regularizer.** Rollouts are from the reference (off-policy w.r.t. current $\theta$). PPO handles this with importance ratios and clipping; mirror descent handles it by assuming the policies are close (the $\ell_2$ term enforces this) and accepting bias in the baseline. Works when the per-iteration update is small.
- **Opens the door to DPO-mirror-descent unification.** DPO and this algorithm share the same mirror-descent derivation structure (Gibbs optimum, log-ratio identity). DPO is the offline-preference-data special case; online mirror descent is the online-reward special case. Understanding one helps with the other.
- **Canonical reference for frontier reasoning RL.** Kimi k1.5 is the only widely-cited contemporary alternative to GRPO for long-CoT RL at scale. If the field branches, this is one of the two trunks.

---

## Gotchas & tricks

- **The "$Z(x) \to$ empirical mean" approximation breaks for small $\tau$.** The identity $\tau \log Z \to \mathbb{E}[r]$ only holds as $\tau \to \infty$. For small $\tau$ (strong regularization), the mean-reward baseline biases the centered reward. The paper uses the empirical mean despite this, because it works — but the approximation is load-bearing. If you tune $\tau$ small, revisit.
- **Optimizer reset matters.** Carrying Adam state across iterations conflates two different optimization problems (different references, different solutions). Don't skip this.
- **$k$ rollouts per problem is undisclosed.** The paper uses $k$ symbolically throughout. Open implementations default to $k = 8\text{--}16$.
- **$\ell_2$ regularizer weight $\tau/2$ is the KL strength.** Large $\tau$ → strong regularization → small per-iteration updates. Small $\tau$ → weak regularization → large updates but potentially unstable (reference-iterate stale). Tune like PPO's $\beta$.
- **Rollouts are fully off-policy within an iteration.** Unlike GRPO (which re-samples between inner mini-batch updates), mirror descent uses the same $k$ rollouts from $\pi_{\theta_i}$ for every gradient step in an iteration. The $\ell_2$ regularizer is supposed to keep this safe, but many inner steps per iteration can push $\pi_\theta$ far from the rollouts and invalidate the regression target. Keep inner steps per iteration small.
- **No clipping means no hard upper bound on per-step update size.** The regularizer is soft; a bad mini-batch can push $\pi_\theta$ far. Monitor $\log(\pi_\theta / \pi_{\theta_i})$ magnitudes during training.
- **Length penalty interacts.** k1.5 adds a length-penalty reward term (see [length-penalty](length-penalty.md)) *on top* of the outcome reward before computing $\bar{r}$ and $r - \bar{r}$. The $\ell_2$ regression still applies to the combined reward.
- **No explicit per-token advantage.** Like GRPO, mirror descent assigns the same per-response reward to every token. For verifiable outcome rewards this is fine; for token-level reward signals (PRM-style), mirror descent would need adaptation.
- **Explicit contrast with R1 is not in the paper.** The paper doesn't name R1 / GRPO / DeepSeekMath anywhere in the text (Jan 2025 simultaneity). Any claims about "Kimi k1.5 is better/worse than R1 at equal compute" are external interpretations.

---

## Sources

- Paper: *Kimi k1.5: Scaling Reinforcement Learning with LLMs* — Moonshot AI (Kimi Team), 2025, arXiv 2501.12599 — introduces the algorithm; Sec. 2.3.1 (setup), Sec. 2.3.2 (derivation + gradient Eq. 3).
- Paper: *Direct Preference Optimization* — Rafailov et al., 2023, arXiv 2305.18290 — same closed-form optimum, used for preferences instead of online rewards.
- Paper: *DeepSeekMath* — Shao et al., 2024, arXiv 2402.03300 — introduces [GRPO](../grpo.md), the value-network-free PPO cousin this algorithm sits alongside.
- Paper: *Proximal Policy Optimization* — Schulman et al., 2017, arXiv 1707.06347 — [PPO](../ppo.md), the baseline this algorithm replaces.
- Textbook: *Convex Optimization* — Boyd & Vandenberghe, 2004 — mirror descent (Section 11.4).
