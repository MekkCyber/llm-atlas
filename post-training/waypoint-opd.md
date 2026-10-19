# Waypoint-OPD
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A gated variant of on-policy distillation that transfers the *RL-induced shift* in a teacher's policy — not its endpoint. For each token, Waypoint-OPD compares the teacher's logit delta at an intermediate RL checkpoint ("waypoint") with its final delta: tokens where the two agree in sign are kept; tokens where they disagree are zeroed out of the student's loss. The result is a distilled student that inherits the teacher's RL gains without copying its RL-era noise or oscillations.

**Prereqs:** [on-policy-distillation.md](./on-policy-distillation.md), [grpo.md](./grpo.md), [rlvr.md](./rlvr.md)
**Related:** [_rl.md](./_rl.md), [rejection-sampling.md](./rejection-sampling.md)

---

## What it is

A teacher model $\pi_T$ is trained by RL, yielding checkpoints $\pi_T^0$ (pre-RL, SFT), $\pi_T^{T/2}$ (intermediate waypoint), and $\pi_T^T$ (post-RL). The information Waypoint-OPD wants to transfer to the student is the *direction* in which RL moved each token's log-probability.

Define per-token deltas:

$$
\Delta_{\text{full}}(x) = \log \pi_T^T(x) - \log \pi_T^0(x), \quad \Delta_{\text{way}}(x) = \log \pi_T^{T/2}(x) - \log \pi_T^0(x)
$$

Waypoint-OPD trains the student only on tokens where $\mathrm{sign}(\Delta_{\text{full}}) = \mathrm{sign}(\Delta_{\text{way}})$ — i.e., where the RL trajectory moved monotonically on this token. The rest are considered unreliable (sign-flipping tokens are often RL noise, oscillation around a plateau, or over-fitting artifacts).

---

## How it works

1. Keep three teacher checkpoints: the pre-RL SFT base, a mid-training waypoint, and the post-RL final.
2. Student samples a batch of completions (on-policy, like OPD).
3. For every student token, compute $\Delta_{\text{full}}$ and $\Delta_{\text{way}}$ against the SFT base.
4. Build a gating mask $m(x) = \mathbb{1}[\, \Delta_{\text{full}} \cdot \Delta_{\text{way}} > 0 \,]$.
5. The distillation loss is the standard OPD KL $\mathrm{KL}(\pi_T^T \,\|\, \pi_\theta)$ **restricted to gated tokens**:

$$
\mathcal{L}_{\text{Waypoint-OPD}} = \frac{\sum_x m(x) \cdot \mathrm{KL}_x(\pi_T^T \,\|\, \pi_\theta)}{\sum_x m(x)}
$$

A small KL anchor to the student's own SFT base is retained for stability.

---

## Why it matters

- **Transfers RL gains, not RL noise.** The waypoint signal filters tokens where the teacher's RL trajectory never settled — exactly the places where naïve endpoint distillation picks up spurious changes.
- **Model-size agnostic.** Reported transfers: Qwen3-4B teacher → Qwen3-8B, Qwen3-14B, Qwen3-30B-A3B students; Waypoint-OPD beats endpoint-only distillation and vanilla OPD on math and code reasoning benchmarks.
- **Cheaper than redoing RL.** You run RL once at a convenient teacher size, then transfer the *shift* to every other size you want to ship, without rerunning the rollout pipeline.
- **Generalizes a diagnostic question.** The sign-agreement test is a reusable probe for "did my RL run actually learn this token, or is it drifting?" — useful beyond distillation.

---

## Gotchas & tricks

- **Waypoint position matters.** Too early (steps near SFT) and nothing has shifted; too late (near endpoint) and the mask approaches identity. The paper uses $T/2$; retuning may be needed for very long RL runs.
- **Only transfers what RL already moved.** Tokens that RL didn't change are gated out — this is on purpose, but if your student is weaker than the teacher's *base*, you still need standard SFT or OPD on top.
- **Requires keeping intermediate checkpoints.** Trivial for new runs; retroactively hard if you threw them away.
- **Sign-agreement is a coarse filter.** Future variants could weight tokens by magnitude agreement or variance; the paper demonstrates the discrete sign-gate is already a strong signal.

---

## Sources

- Paper: *Waypoint-OPD: Transferring RL-Induced Policy Shifts via Gated On-Policy Distillation* — Jiang, Quan, Li, Feng, Liu, Zhang, Alibaba Cloud, 2026 — arXiv [2610.21601](https://arxiv.org/abs/2610.21601).
- Related: *DeepSeek-R1* R1-Distill recipe — baseline for endpoint-only transfer.
