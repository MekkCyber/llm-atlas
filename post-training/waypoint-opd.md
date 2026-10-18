# Waypoint-OPD
*Depth — gated on-policy distillation using a mid-RL teacher checkpoint.*

**TL;DR:** Standard on-policy distillation copies the full base→final policy shift of an RL-trained teacher into a smaller (or re-initialized) student. Waypoint-OPD saves an intermediate "waypoint" checkpoint, decomposes the shift into early (base→waypoint) and late (waypoint→final) halves, and *gates* the endpoint shift at each student token by whether the two halves agree in direction. Agreement → full transfer; disagreement → damped transfer. One extra frozen scoring pass per step.

**Prereqs:** [grpo.md](grpo.md), [rlvr.md](rlvr.md), [_rl.md](_rl.md)
**Related:** [rejection-sampling.md](rejection-sampling.md), [ppo.md](ppo.md)

---

## What it is

A targeted fix for noisy distillation signal in RLVR teachers. Late-stage RL can overshoot or chase reward-hack directions; distilling the raw endpoint drags the student into those directions too. Waypoint-OPD uses the teacher's own training trajectory as a sanity check on each component of the final shift.

## How it works

Keep three teachers: the base `π_base`, a mid-RL waypoint `π_wp`, and the endpoint `π_end`. For a student rollout token with context `s`:

1. Compute per-token log-ratios `Δ_early = log π_wp(·|s) − log π_base(·|s)` and `Δ_late = log π_end(·|s) − log π_wp(·|s)`.
2. Form a gate `g ∝ sign(Δ_early · Δ_late)·min(|Δ_early|,|Δ_late|)` (direction + magnitude agreement).
3. Distill student toward a *gated* target `log π_base + g · (Δ_early + Δ_late)` instead of `log π_end`.

Only one extra forward pass is needed since the three teacher log-probs are read from frozen snapshots.

## Why it matters

Teacher-shift gating lets you reuse a shelved RL run that produced *both* a waypoint and endpoint — common when training saves checkpoints — without re-collecting rollouts for a new distillation pipeline. It's an on-policy method (gates are evaluated on student rollouts), so it stays compatible with existing OPD infra.

## Gotchas & tricks

- Requires the waypoint to be meaningful; using the base-itself gives `g≈0`. Pick a waypoint partway through training.
- Gating increases *bias* toward "safe" shifts and can underfit newly-emergent behaviors; raise the gate temperature if the student lags.
- Reported wins are on math/code with Qwen3-8B/14B/30B-A3B students — generalization to non-verifiable rewards is untested.

## Sources

- Paper: *Waypoint-OPD: Transferring RL-Induced Policy Shifts via Gated On-Policy Distillation* — Jiang et al., Alibaba Cloud, 2026 — [arXiv:2610.21601](https://arxiv.org/abs/2610.21601)
