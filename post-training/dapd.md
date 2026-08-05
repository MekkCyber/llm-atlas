# DAPD: Dual-Anchored Policy Distillation
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A fix for the "privilege illusion" in on-policy self-distillation (OPSD): the student learns privileged-context-dependent behaviour that it can't reproduce at inference. DAPD adds two levels of *anchoring* — matched-information paths (DPA) and bidirectional reference↔rollout alignment (DSA) — that keep the student's inference-time behaviour consistent with the training-time reference. +2.00 avg pts over vanilla OPSD on Qwen3-4B; gains persist at 32B.

**Prereqs:** [on-policy-distillation](on-policy-distillation.md), [_rl](_rl.md)
**Related:** [rejection-sampling](rejection-sampling.md), [grpo](grpo.md)

---

## What it is

Standard OPSD samples a student rollout $y$ from $\pi_\theta(\cdot \mid x)$, then supervises with a teacher $\pi_T(\cdot \mid x, z, y_{<t})$ that sees privileged information $z$. The failure mode DAPD names: the student learns to *behave as if $z$ were still available at inference*, which is a false representation of the world and degrades downstream performance.

DAPD attacks this with two anchors:
- **Dual-Path Anchoring (DPA)** — introduce a self-conditioned bridge and match rollout behaviour to a reference path with the same information budget.
- **Dual-Source Anchoring (DSA)** — apply the alignment in *both* reference→rollout and rollout→reference directions.

## How it works

**Dual-Path Anchoring (DPA).** Two per-token distributions are compared:
1. **Reference path**: $\pi_T(\cdot \mid x, z, y_{<t})$ — teacher with privileged $z$ (the OPSD target).
2. **Self-conditioned bridge**: $\pi_\theta(\cdot \mid x, y_{<t})$ *with a matched-information self-conditioning signal* so the student's prediction is anchored to what it *would* have produced given equivalent context, not to a $z$-inflated target.

The loss aligns these two along matched-information paths — the student can't drift toward $z$-dependent behaviour because both paths carry the same information budget by construction.

**Dual-Source Anchoring (DSA).** Symmetric application:
- Reference→Rollout: standard OPSD direction (teacher supervises student).
- Rollout→Reference: student's on-policy prediction is projected onto the reference to reduce dependence on privileged reference guidance while preserving correctness.

The two together give a symmetric, matched-information distillation objective — hence "dual anchor."

## Why it matters

- **Ceiling raise, not a workaround.** The privilege illusion silently caps how much OPSD can help. DAPD lifts that cap without adding new hyperparameters at scale (author reports gains at both 4B and 32B).
- **Cheap.** Same rollout budget as OPSD; the extra forward pass for the bridge is small.
- **Composable.** DAPD is a loss and rollout-scheme modification. Any base-model + teacher setup that runs OPSD can adopt it.

## Gotchas & tricks

- **The bridge must be genuinely matched-information.** If the self-conditioning signal leaks any of $z$, DAPD collapses back to OPSD.
- **DSA weighting.** The two directions (ref→rollout, rollout→ref) shouldn't be equal — the paper's default weighting favours reference→rollout, using rollout→reference as a soft regulariser.
- **Watch for teacher-collapse.** As the student improves, teacher-student gap shrinks; DSA can start pushing the teacher's own signal into noise. Freeze rollout→reference after a warmup, or decay its weight.
- **Diagnostic: measure inference-time behaviour under $z$-absent conditions during training.** If the student is still improving on the training loss but degrading on $z$-absent eval, the anchoring is not working.

## Sources

- Paper: *DAPD: Dual-Anchored Policy Distillation* — arXiv:2608.01735, 2026 — introduces both anchoring mechanisms and the "privilege illusion" diagnosis.
- Code: [github.com/uanu2002/DAPD](https://github.com/uanu2002/DAPD).
