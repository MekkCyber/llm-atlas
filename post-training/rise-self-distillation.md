# RISE: Self-Extrapolating Policy Distillation
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** On-policy distillation (OPD) provides dense per-token supervision but is bottlenecked by teacher quality: external teachers are distribution-mismatched, self-distillation with privileged conditioning is capped by in-context capacity. **RISE** synthesizes a teacher **from the model's own training trajectory** by extrapolating parameter- or logit-space differences between two snapshots, converting sparse outcome-based reward signals into dense token-level supervision — then loops.

**Prereqs:** [_rl.md](./_rl.md), [rlvr.md](./rlvr.md)
**Related:** [rejection-sampling.md](./rejection-sampling.md), [reasoning/long-cot-rl.md](./reasoning/long-cot-rl.md), [grpo.md](./grpo.md)

---

## What it is

Two ways to give an RL-trained student model dense supervision:

- **External teacher OPD.** Distill from a bigger model. Problem: distribution mismatch — the teacher visits states the student cannot, and vice-versa.
- **Self-distillation with privileged conditioning.** Teacher = student conditioned on an oracle hint. Problem: capped by in-context learning capacity; the hint often does not translate to the student's parameters.

RISE observes: over an RL trajectory the model *is* improving. If you can extract that improvement direction and *project it forward*, the projected checkpoint is a better teacher than any snapshot on the trajectory — and it comes entirely from the model's own history.

## How it works

- **Take two RL snapshots.** $\theta_t$ (later) and $\theta_{t-k}$ (earlier).
- **Extract the improvement direction.**
  - **Parameter-space:** $\Delta\theta = \theta_t - \theta_{t-k}$.
  - **Logit-space:** at each state $s$, $\Delta \ell(s) = \ell_{\theta_t}(s) - \ell_{\theta_{t-k}}(s)$.
- **Extrapolate.** Synthetic teacher $= \theta_t + \alpha \cdot \Delta\theta$ (parameter mode) or $\ell(s) + \alpha \cdot \Delta\ell(s)$ (logit mode), for some step $\alpha > 0$.
- **On-policy distill the student to match this synthetic teacher token-by-token.** This is where sparse outcome-based updates get *densified* — every token now carries a supervision signal from the projected-forward teacher's distribution.
- **Loop.** Combine the RL step (value/verifiable reward) with the OPD step in a recursive schedule: train → snapshot → extrapolate → distill → repeat.

## Why it matters

- **Dense signal from sparse reward.** Turns terminal correct/incorrect rewards into per-token targets without any external model — attractive for domains where frontier teachers are unavailable or expensive.
- **Distribution match is automatic.** The synthetic teacher lives in the student's parameter neighborhood, so the states it visits overlap the student's own — no mismatch tax.
- **Recursive self-improvement without external supervision.** The regime frontier post-training is stuck in.
- **Cross-domain gains.** Reported improvements over OPD baselines on math reasoning, STEM, code generation, and multi-turn tasks.

## Gotchas & tricks

- **Extrapolation step $\alpha$ is fragile.** Too small = teacher ≈ student, no signal; too large = teacher moves into an incoherent parameter region. Tune per-run.
- **Parameter-mode vs. logit-mode.** Parameter extrapolation is cheaper (one delta per checkpoint pair) but more sensitive to noise; logit extrapolation costs an extra forward per state but is more stable.
- **Requires a stable RL trajectory.** If the underlying RL is noisy (large step-to-step reward variance), the extracted direction is dominated by noise and RISE degenerates.
- **Interacts with KL to reference.** The synthetic teacher is *not* the reference model — do not mix them into the same regularizer. RISE's OPD term is the dense-signal channel; keep RL's KL-to-reference term separate.
- **Beware feedback cycles.** RISE loops synthetic supervision back into the same trajectory. Snapshot cadence and $\alpha$ schedule need to prevent the extrapolation from tracking its own artifacts.

## Sources

- Paper: *RISE: Recursive Improvement via Self-Extrapolating Policy Distillation* — Yang Li, Semih Yavuz, Shafiq Joty — Salesforce Research — arXiv:2609.05295 (2026).
