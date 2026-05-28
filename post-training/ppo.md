# Proximal Policy Optimization (PPO)
*Depth — the clipped-ratio policy-gradient algorithm that became the default for classical RLHF.*

**TL;DR:** A first-order policy-gradient algorithm that keeps the policy in a trust region by **clipping the importance ratio** $\pi_\theta / \pi_{\theta_{\text{old}}}$ to $[1-\epsilon, 1+\epsilon]$ (default $\epsilon = 0.2$). Sample a batch with the old policy, do $K$ epochs of SGD on a clipped surrogate objective, repeat. Cheaper and simpler than TRPO (no conjugate-gradient solve, no Fisher-vector products); strictly a heuristic approximation of TRPO's monotonic-improvement guarantee. Became the default algorithm for classical RLHF (InstructGPT, Claude, GPT-4 post-training) and the direct ancestor of GRPO.

**Prereqs:** [_rl](_rl.md)
**Related:** [grpo](grpo.md) · [dpo](dpo.md) · [online-policy-mirror-descent](reasoning/online-policy-mirror-descent.md) · [rlvr](rlvr.md) · [_rewards](_rewards.md)

---

## What it is

A policy-gradient algorithm that answers three questions:

1. **How do we estimate the advantage?** — GAE (Generalized Advantage Estimation) with a learned value network.
2. **How do we keep the policy close to the rollouts it came from?** — clip the importance ratio at $[1-\epsilon, 1+\epsilon]$.
3. **How do we get data efficiency?** — run $K$ epochs of SGD on each batch of rollouts before collecting new ones.

PPO sits between vanilla policy gradient (one gradient step per batch, unstable) and TRPO (hard KL constraint with second-order machinery, complex). It trades TRPO's monotonic-improvement theorem for a loss-shaped heuristic that is easier to implement and composes with architectures that have dropout, layer sharing, or auxiliary heads.

---

## How it works

### The importance ratio

For a trajectory collected under the old policy $\pi_{\theta_{\text{old}}}$, define:

$$
r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}
$$

At $\theta = \theta_{\text{old}}$, $r_t = 1$. As the new policy diverges from the old one, $r_t$ drifts away from 1. PPO's whole idea is to **bound how far this drift can go** before the loss signal turns off.

### The naive (unclipped) surrogate — "CPI"

Standard importance-weighted policy gradient:

$$
L^{\text{CPI}}(\theta) = \mathbb{E}_t [\, r_t(\theta) \cdot A_t \,] \quad \text{(Schulman 2017, Eq. 6)}
$$

where $A_t$ is the advantage estimate (see GAE below). Maximizing $L^{\text{CPI}}$ is a valid surrogate for the expected return near $\theta_{\text{old}}$, but nothing prevents the optimizer from pushing $r_t$ far from 1 — which makes the surrogate an unreliable proxy for the true objective.

### The clipped surrogate

The main PPO objective:

$$
L^{\text{CLIP}}(\theta) = \mathbb{E}_t \!\left[ \min\!\left( r_t(\theta) \cdot A_t,\; \mathrm{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \cdot A_t \right) \right] \quad \text{(Eq. 7)}
$$

$\mathrm{clip}(x, a, b) = \max(a, \min(x, b))$. The $\min(\ldots)$ takes the *smaller* of the unclipped and clipped terms — making $L^{\text{CLIP}}$ a **pessimistic lower bound** on $L^{\text{CPI}}$. Read by cases:

- **$A_t > 0$** (action was good): the objective rewards increasing $r_t$, but once $r_t > 1+\epsilon$ the clipped term kicks in and the gradient is zero. No incentive to push further.
- **$A_t < 0$** (action was bad): the objective rewards decreasing $r_t$, but once $r_t < 1-\epsilon$ the gradient is zero. No incentive to push further in the other direction.
- In both cases, if moving outside the $[1-\epsilon, 1+\epsilon]$ band would make the objective *worse*, the unclipped term is still active — so bad actions are always penalized, even past the clip.

The clip is asymmetric in the direction that matters: **pessimistic either way**.

Default $\epsilon = 0.2$ (Schulman 2017, MuJoCo runs). $\epsilon = 0.1$ is also used (Atari).

### GAE — the advantage estimator

PPO's advantage comes from a learned value network $V(s)$ plus truncated n-step bootstrapping. The per-step TD residual:

$$
\delta_t = r_t + \gamma \cdot V(s_{t+1}) - V(s_t)
$$

(Here $r_t$ is the reward at step $t$ — not the importance ratio; notation collision in the paper.) The GAE advantage is an exponentially-weighted sum of $\delta$'s:

$$
A_t = \delta_t + (\gamma\lambda)\, \delta_{t+1} + (\gamma\lambda)^2 \delta_{t+2} + \ldots + (\gamma\lambda)^{T-t+1}\, \delta_{T-1} \quad \text{(Eq. 11)}
$$

$\lambda$ trades bias vs variance: $\lambda = 1$ → pure Monte-Carlo (high variance, low bias); $\lambda = 0$ → one-step TD (low variance, high bias). Typical default $\lambda = 0.95$, $\gamma = 0.99$.

### The combined loss — value + entropy

When the policy and value networks share parameters (common in Atari-sized architectures, uncommon for LLMs), PPO combines three terms:

$$
L^{\text{CLIP+VF+S}}(\theta) = \mathbb{E}_t \!\left[ L^{\text{CLIP}}(\theta) - c_1 \cdot L^{\text{VF}}(\theta) + c_2 \cdot S[\pi_\theta](s_t) \right] \quad \text{(Eq. 9)}
$$

- **Value-function loss**: $L^{\text{VF}} = (V_\theta(s_t) - V_t^{\text{target}})^2$ — the value network regresses onto returns.
- **Entropy bonus**: $S[\pi_\theta]$ = entropy of the policy distribution. Encourages exploration.
- **Coefficients**: $c_1 = 1$, $c_2 = 0.01$ (Atari defaults). LLM-scale RL often sets $c_1 = 0$ (separate value net), $c_2 = 0$ (KL penalty handles exploration).

### The adaptive-KL variant

PPO has a less-famous second variant that replaces the clip with an explicit KL penalty:

$$
L^{\text{KLPEN}}(\theta) = \mathbb{E}_t \!\left[ r_t(\theta) \cdot A_t - \beta \cdot \mathrm{KL}( \pi_{\theta_{\text{old}}} \,\|\, \pi_\theta ) \right] \quad \text{(Eq. 8)}
$$

$\beta$ is adapted after each update based on the measured KL:

```
d = E_t [ KL( π_θ_old || π_θ ) ]

if d < d_target / 1.5:  β ← β / 2        (policy moved too little, loosen)
if d > d_target · 1.5:  β ← β · 2        (policy moved too much, tighten)
```

Typical $d_{\text{target}} \in [0.003, 0.03]$. The paper says the algorithm is "not very sensitive" to the 1.5 / 2 heuristics. In their MuJoCo results (Table 1), clipping ($\epsilon = 0.2$) scored 0.82 vs 0.74 for adaptive-KL. **Most practical RLHF pipelines use the clip variant plus a separate KL-to-reference penalty** (see below).

### The algorithm

```
for iteration = 1..N:
    for actor = 1..N_actors in parallel:
        collect trajectory of length T using π_θ_old
        compute A_1..A_T via GAE using V_θ_old
    for epoch = 1..K:
        for minibatch in all N_actors × T samples:
            gradient step on L^CLIP+VF+S
    θ_old ← θ        # next iteration uses the updated policy for rollouts
```

Typical hyperparameters (MuJoCo defaults from the paper):

| Knob | Value |
|---|---|
| Horizon $T$ | 2048 |
| Epochs $K$ | 10 |
| Minibatch size | 64 |
| Discount $\gamma$ | 0.99 |
| GAE $\lambda$ | 0.95 |
| Clip $\epsilon$ | 0.2 |
| Adam LR | $3 \times 10^{-4}$ |

### PPO for LLM RLHF — the InstructGPT pattern

The classical RLHF pipeline uses PPO with specific LLM adaptations:

$$
L_{\text{RLHF}}(\theta) = \mathbb{E}_{q, o} \!\left[ \min( r \cdot A,\; \mathrm{clip}(r, 1-\epsilon, 1+\epsilon) \cdot A ) \right] - \beta \cdot \mathbb{E}_{q, o} \!\left[ \mathrm{KL}( \pi_\theta(\,\cdot\, \mid q, o_{<t}) \,\|\, \pi_{\text{ref}}(\,\cdot\, \mid q, o_{<t}) ) \right] - c_1 \cdot \mathbb{E} \!\left[ (V_\theta(s) - V_{\text{target}})^2 \right]
$$

Differences from the classical RL setting:

- **Terminal reward**, not per-step. The preference reward model scores the full response; $r$ is applied at the last token (or broadcast to all response tokens).
- **KL penalty to a frozen reference** $\pi_{\text{ref}}$ (usually the SFT checkpoint) — a *separate* term from the PPO clip. The clip controls drift from $\pi_{\theta_{\text{old}}}$ within a single iteration; the KL controls drift from $\pi_{\text{ref}}$ globally. Both are needed: without the KL, the policy reward-hacks; without the clip, single updates overshoot.
- **Value network is a separate copy of the policy** with a scalar head. For frontier LLMs this doubles RL compute (a second ~70B-100B forward/backward per step).
- **Typical $\beta \in [0.001, 0.1]$**. Too low → reward hacking. Too high → policy can't move.

---

## Why it matters

- **Simpler than TRPO.** No conjugate gradient, no Fisher-vector products, no second-order approximation. First-order SGD only. Compatible with dropout, shared trunks, and auxiliary heads.
- **Data-efficient.** $K = 10$ epochs per batch vs 1 for vanilla policy gradient. The clip makes re-using the same rollouts for multiple optimization steps safe.
- **Robust across tasks.** Near-default hyperparameters work on MuJoCo, Atari, and LLM post-training with modest tuning. The paper's pitch is exactly this — "the data efficiency and reliable performance of TRPO with only first-order optimization."
- **The foundation of modern LLM RL.** Every mainstream modern variant is "PPO with X removed": GRPO (no value net), k1.5 mirror descent (no clip, $\ell_2$ regularizer instead), DPO (no rollouts). Understanding PPO is how you understand what each variant is trading.
- **Still production RLHF.** InstructGPT, early Claude, GPT-4 post-training all use PPO with preference reward models. When people say "RLHF" without qualification, they usually mean PPO-RLHF.

---

## Gotchas & tricks

- **The $K$ epochs are off-policy after epoch 1.** Epoch 1 is on-policy; by epoch 10, $\pi_\theta$ has drifted from $\pi_{\theta_{\text{old}}}$ and the importance ratio is no longer $\approx 1$. The clip is what keeps this safe, but if $K$ is too large the clip fires on most samples and further updates are no-ops. Typical $K = 3\text{--}10$.
- **Value network is expensive for LLMs.** At LLM scale, the critic approximately doubles RL compute. Modern reasoning RL drops it (GRPO, mirror descent). For preference-RM RLHF you usually still want it — the reward model is trained, so a learned value baseline is informative.
- **KL-to-ref vs clip are different things.** The clip is an *intra-iteration* trust region (don't overshoot within one gradient step on this batch). The KL penalty to $\pi_{\text{ref}}$ is a *global* trust region (don't drift too far from the SFT checkpoint across training). They both matter; conflating them is a common source of RLHF instability.
- **Reward normalization / advantage normalization.** Common unrecorded trick: normalize advantages to mean 0, std 1 within each minibatch. The paper doesn't mention it but every open implementation does it. Not doing it makes PPO sensitive to reward scale.
- **Value-loss clipping.** Similarly common: clip the value-function loss like the policy loss. Not in the paper. Reduces instability when $V_\theta$ drifts between rollout collection and update.
- **Entropy bonus is not free.** Too much entropy → the policy stays random; too little → policy collapses. For LLMs the KL-to-ref term usually handles exploration; an explicit entropy bonus is rare.
- **Per-token vs per-trajectory advantages.** Classical PPO computes advantages per-step using the value net. For LLM RLHF with a terminal reward, people often broadcast the terminal advantage to all response tokens (with discounting). Be explicit about which convention you use; implementations differ.
- **Adaptive-KL vs clip.** The paper's ablation favored clip. In practice for LLMs, people use clip for the intra-iteration trust region and a *separate adaptive or fixed* KL to the reference model. The "adaptive-KL PPO" of the original paper is almost never used at scale.
- **PPO's trust region is a heuristic, not a theorem.** TRPO has a monotonic-improvement theorem with a KL constraint. PPO gives up the theorem for implementation simplicity. For most practical problems this is fine; for theoretically sensitive tasks (safety-critical RL), understand what you're trading.
- **Minibatch size interacts with clip.** Smaller minibatches → noisier updates → more clip firings per sample → effectively less off-policy exploitation. Larger minibatches average out more, so fewer clips trigger. Tune together.
- **Frontier RLHF hyperparameters are mostly undisclosed.** InstructGPT gave rough numbers; later papers (GPT-4, Claude) don't. Open implementations (TRL, DeepSpeed-Chat) default to $\epsilon = 0.2$, $K = 1\text{--}4$, $\beta_{\text{KL}} = 0.01\text{--}0.05$. Treat these as starting points.

---

## Sources

- Paper: *Proximal Policy Optimization Algorithms* — Schulman, Wolski, Dhariwal, Radford, Klimov, OpenAI, 2017, arXiv 1707.06347 — the PPO objective, both clipped and adaptive-KL variants.
- Paper: *High-Dimensional Continuous Control Using Generalized Advantage Estimation* — Schulman et al., 2015, arXiv 1506.02438 — GAE, the advantage estimator PPO inherits.
- Paper: *Trust Region Policy Optimization* — Schulman et al., 2015, arXiv 1502.05477 — TRPO, PPO's direct predecessor.
- Paper: *Training language models to follow instructions with human feedback (InstructGPT)* — Ouyang et al., 2022, arXiv 2203.02155 — canonical PPO-RLHF applied to LLMs.
- Paper: *Fine-Tuning Language Models from Human Preferences* — Ziegler et al., 2019, arXiv 1909.08593 — the earlier PPO-on-LM paper that InstructGPT builds on.
