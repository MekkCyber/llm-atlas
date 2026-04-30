# Proximal Policy Optimization (PPO)
*Depth — the clipped-ratio policy-gradient algorithm that became the default for classical RLHF.*

**TL;DR:** A first-order policy-gradient algorithm that keeps the policy in a trust region by **clipping the importance ratio** `π_θ / π_θ_old` to `[1-ε, 1+ε]` (default `ε = 0.2`). Sample a batch with the old policy, do `K` epochs of SGD on a clipped surrogate objective, repeat. Cheaper and simpler than TRPO (no conjugate-gradient solve, no Fisher-vector products); strictly a heuristic approximation of TRPO's monotonic-improvement guarantee. Became the default algorithm for classical RLHF (InstructGPT, Claude, GPT-4 post-training) and the direct ancestor of GRPO.

**Prereqs:** [_rl](_rl.md)
**Related:** [grpo](grpo.md) · [dpo](dpo.md) · [online-policy-mirror-descent](reasoning/online-policy-mirror-descent.md) · [rlvr](rlvr.md) · [_rewards](_rewards.md)

---

## What it is

A policy-gradient algorithm that answers three questions:

1. **How do we estimate the advantage?** — GAE (Generalized Advantage Estimation) with a learned value network.
2. **How do we keep the policy close to the rollouts it came from?** — clip the importance ratio at `[1-ε, 1+ε]`.
3. **How do we get data efficiency?** — run `K` epochs of SGD on each batch of rollouts before collecting new ones.

PPO sits between vanilla policy gradient (one gradient step per batch, unstable) and TRPO (hard KL constraint with second-order machinery, complex). It trades TRPO's monotonic-improvement theorem for a loss-shaped heuristic that is easier to implement and composes with architectures that have dropout, layer sharing, or auxiliary heads.

---

## How it works

### The importance ratio

For a trajectory collected under the old policy `π_θ_old`, define:

```
r_t(θ) = π_θ(a_t | s_t) / π_θ_old(a_t | s_t)
```

At `θ = θ_old`, `r_t = 1`. As the new policy diverges from the old one, `r_t` drifts away from 1. PPO's whole idea is to **bound how far this drift can go** before the loss signal turns off.

### The naive (unclipped) surrogate — "CPI"

Standard importance-weighted policy gradient:

```
L^CPI(θ) = E_t [ r_t(θ) · A_t ]                        (Schulman 2017, Eq. 6)
```

where `A_t` is the advantage estimate (see GAE below). Maximizing `L^CPI` is a valid surrogate for the expected return near `θ_old`, but nothing prevents the optimizer from pushing `r_t` far from 1 — which makes the surrogate an unreliable proxy for the true objective.

### The clipped surrogate

The main PPO objective:

```
L^CLIP(θ) = E_t [ min( r_t(θ) · A_t,
                       clip(r_t(θ), 1-ε, 1+ε) · A_t ) ]   (Eq. 7)
```

`clip(x, a, b) = max(a, min(x, b))`. The `min(...)` takes the *smaller* of the unclipped and clipped terms — making `L^CLIP` a **pessimistic lower bound** on `L^CPI`. Read by cases:

- **`A_t > 0`** (action was good): the objective rewards increasing `r_t`, but once `r_t > 1+ε` the clipped term kicks in and the gradient is zero. No incentive to push further.
- **`A_t < 0`** (action was bad): the objective rewards decreasing `r_t`, but once `r_t < 1-ε` the gradient is zero. No incentive to push further in the other direction.
- In both cases, if moving outside the `[1-ε, 1+ε]` band would make the objective *worse*, the unclipped term is still active — so bad actions are always penalized, even past the clip.

The clip is asymmetric in the direction that matters: **pessimistic either way**.

Default `ε = 0.2` (Schulman 2017, MuJoCo runs). `ε = 0.1` is also used (Atari).

### GAE — the advantage estimator

PPO's advantage comes from a learned value network `V(s)` plus truncated n-step bootstrapping. The per-step TD residual:

```
δ_t = r_t + γ · V(s_{t+1}) - V(s_t)
```

(Here `r_t` is the reward at step t — not the importance ratio; notation collision in the paper.) The GAE advantage is an exponentially-weighted sum of δ's:

```
A_t = δ_t + (γλ) δ_{t+1} + (γλ)^2 δ_{t+2} + … + (γλ)^{T-t+1} δ_{T-1}   (Eq. 11)
```

`λ` trades bias vs variance: `λ = 1` → pure Monte-Carlo (high variance, low bias); `λ = 0` → one-step TD (low variance, high bias). Typical default `λ = 0.95`, `γ = 0.99`.

### The combined loss — value + entropy

When the policy and value networks share parameters (common in Atari-sized architectures, uncommon for LLMs), PPO combines three terms:

```
L^CLIP+VF+S(θ) = E_t [ L^CLIP(θ) - c_1 · L^VF(θ) + c_2 · S[π_θ](s_t) ]   (Eq. 9)
```

- **Value-function loss**: `L^VF = (V_θ(s_t) - V_t^target)^2` — the value network regresses onto returns.
- **Entropy bonus**: `S[π_θ]` = entropy of the policy distribution. Encourages exploration.
- **Coefficients**: `c_1 = 1, c_2 = 0.01` (Atari defaults). LLM-scale RL often sets `c_1 = 0` (separate value net), `c_2 = 0` (KL penalty handles exploration).

### The adaptive-KL variant

PPO has a less-famous second variant that replaces the clip with an explicit KL penalty:

```
L^KLPEN(θ) = E_t [ r_t(θ) · A_t - β · KL( π_θ_old || π_θ ) ]           (Eq. 8)
```

`β` is adapted after each update based on the measured KL:

```
d = E_t [ KL( π_θ_old || π_θ ) ]

if d < d_target / 1.5:  β ← β / 2        (policy moved too little, loosen)
if d > d_target · 1.5:  β ← β · 2        (policy moved too much, tighten)
```

Typical `d_target ∈ [0.003, 0.03]`. The paper says the algorithm is "not very sensitive" to the 1.5 / 2 heuristics. In their MuJoCo results (Table 1), clipping (ε=0.2) scored 0.82 vs 0.74 for adaptive-KL. **Most practical RLHF pipelines use the clip variant plus a separate KL-to-reference penalty** (see below).

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
| Horizon T | 2048 |
| Epochs K | 10 |
| Minibatch size | 64 |
| Discount γ | 0.99 |
| GAE λ | 0.95 |
| Clip ε | 0.2 |
| Adam LR | 3 × 10⁻⁴ |

### PPO for LLM RLHF — the InstructGPT pattern

The classical RLHF pipeline uses PPO with specific LLM adaptations:

```
L_RLHF(θ) = E_{q, o} [ min( r · A, clip(r, 1-ε, 1+ε) · A ) ]
          - β · E_{q, o} [ KL( π_θ( · | q, o_<t) || π_ref( · | q, o_<t) ) ]
          - c_1 · E [ (V_θ(s) - V_target)^2 ]
```

Differences from the classical RL setting:

- **Terminal reward**, not per-step. The preference reward model scores the full response; `r` is applied at the last token (or broadcast to all response tokens).
- **KL penalty to a frozen reference** `π_ref` (usually the SFT checkpoint) — a *separate* term from the PPO clip. The clip controls drift from `π_θ_old` within a single iteration; the KL controls drift from `π_ref` globally. Both are needed: without the KL, the policy reward-hacks; without the clip, single updates overshoot.
- **Value network is a separate copy of the policy** with a scalar head. For frontier LLMs this doubles RL compute (a second `~70B-100B` forward/backward per step).
- **Typical `β ∈ [0.001, 0.1]`**. Too low → reward hacking. Too high → policy can't move.

---

## Why it matters

- **Simpler than TRPO.** No conjugate gradient, no Fisher-vector products, no second-order approximation. First-order SGD only. Compatible with dropout, shared trunks, and auxiliary heads.
- **Data-efficient.** `K = 10` epochs per batch vs 1 for vanilla policy gradient. The clip makes re-using the same rollouts for multiple optimization steps safe.
- **Robust across tasks.** Near-default hyperparameters work on MuJoCo, Atari, and LLM post-training with modest tuning. The paper's pitch is exactly this — "the data efficiency and reliable performance of TRPO with only first-order optimization."
- **The foundation of modern LLM RL.** Every mainstream modern variant is "PPO with X removed": GRPO (no value net), k1.5 mirror descent (no clip, ℓ₂ regularizer instead), DPO (no rollouts). Understanding PPO is how you understand what each variant is trading.
- **Still production RLHF.** InstructGPT, early Claude, GPT-4 post-training all use PPO with preference reward models. When people say "RLHF" without qualification, they usually mean PPO-RLHF.

---

## Gotchas & tricks

- **The `K` epochs are off-policy after epoch 1.** Epoch 1 is on-policy; by epoch 10, `π_θ` has drifted from `π_θ_old` and the importance ratio is no longer ≈ 1. The clip is what keeps this safe, but if `K` is too large the clip fires on most samples and further updates are no-ops. Typical `K = 3-10`.
- **Value network is expensive for LLMs.** At LLM scale, the critic approximately doubles RL compute. Modern reasoning RL drops it (GRPO, mirror descent). For preference-RM RLHF you usually still want it — the reward model is trained, so a learned value baseline is informative.
- **KL-to-ref vs clip are different things.** The clip is an *intra-iteration* trust region (don't overshoot within one gradient step on this batch). The KL penalty to `π_ref` is a *global* trust region (don't drift too far from the SFT checkpoint across training). They both matter; conflating them is a common source of RLHF instability.
- **Reward normalization / advantage normalization.** Common unrecorded trick: normalize advantages to mean 0, std 1 within each minibatch. The paper doesn't mention it but every open implementation does it. Not doing it makes PPO sensitive to reward scale.
- **Value-loss clipping.** Similarly common: clip the value-function loss like the policy loss. Not in the paper. Reduces instability when `V_θ` drifts between rollout collection and update.
- **Entropy bonus is not free.** Too much entropy → the policy stays random; too little → policy collapses. For LLMs the KL-to-ref term usually handles exploration; an explicit entropy bonus is rare.
- **Per-token vs per-trajectory advantages.** Classical PPO computes advantages per-step using the value net. For LLM RLHF with a terminal reward, people often broadcast the terminal advantage to all response tokens (with discounting). Be explicit about which convention you use; implementations differ.
- **Adaptive-KL vs clip.** The paper's ablation favored clip. In practice for LLMs, people use clip for the intra-iteration trust region and a *separate adaptive or fixed* KL to the reference model. The "adaptive-KL PPO" of the original paper is almost never used at scale.
- **PPO's trust region is a heuristic, not a theorem.** TRPO has a monotonic-improvement theorem with a KL constraint. PPO gives up the theorem for implementation simplicity. For most practical problems this is fine; for theoretically sensitive tasks (safety-critical RL), understand what you're trading.
- **Minibatch size interacts with clip.** Smaller minibatches → noisier updates → more clip firings per sample → effectively less off-policy exploitation. Larger minibatches average out more, so fewer clips trigger. Tune together.
- **Frontier RLHF hyperparameters are mostly undisclosed.** InstructGPT gave rough numbers; later papers (GPT-4, Claude) don't. Open implementations (TRL, DeepSpeed-Chat) default to `ε = 0.2`, `K = 1-4`, `β_KL = 0.01-0.05`. Treat these as starting points.

---

## Sources

- Paper: *Proximal Policy Optimization Algorithms* — Schulman, Wolski, Dhariwal, Radford, Klimov, OpenAI, 2017, arXiv 1707.06347 — the PPO objective, both clipped and adaptive-KL variants.
- Paper: *High-Dimensional Continuous Control Using Generalized Advantage Estimation* — Schulman et al., 2015, arXiv 1506.02438 — GAE, the advantage estimator PPO inherits.
- Paper: *Trust Region Policy Optimization* — Schulman et al., 2015, arXiv 1502.05477 — TRPO, PPO's direct predecessor.
- Paper: *Training language models to follow instructions with human feedback (InstructGPT)* — Ouyang et al., 2022, arXiv 2203.02155 — canonical PPO-RLHF applied to LLMs.
- Paper: *Fine-Tuning Language Models from Human Preferences* — Ziegler et al., 2019, arXiv 1909.08593 — the earlier PPO-on-LM paper that InstructGPT builds on.
