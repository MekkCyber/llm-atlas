# On-Policy Self-Distillation for AR Video (OPSD-V)
*Depth — one specific technique, grounded in its source paper.*

**TL;DR:** Few-step autoregressive (AR) video diffusion models (Self-Forcing, LongLive) are fast but **drift on long rollouts** — errors accumulate frame by frame, motion degrades. OPSD-V post-trains them via **on-policy self-distillation**: the student model runs the actual inference-time rollout with its stochastic AR cache, a teacher (same architecture) supplies **dense denoising-level correction targets** on the student's own trajectory using a cleaner AR-consistent temporal cache. Sampler and step count are unchanged. Meituan × HKUST, 2026.

**Prereqs:** [../post-training/_post-training.md](../post-training/_post-training.md)
**Related:** [../post-training/rejection-sampling.md](../post-training/rejection-sampling.md) · [../post-training/dpo.md](../post-training/dpo.md)

---

## What it is

Autoregressive video diffusion (predict next frame chunk given the past) is the current recipe for long, low-latency video generation. To hit real-time, models like Self-Forcing and LongLive distill dense diffusion into **few-step samplers** (2–8 denoising steps per frame). This is cheap but leaves the model exposed to **AR-cache drift**: at inference the temporal cache is populated by the model's own noisy outputs, whereas at training it was populated by ground truth. Small errors compound over long rollouts.

OPSD-V is a **post-training pass** that closes this train/inference gap for AR video diffusion.

## How it works

**Two-stream setup.**

*Student.* Runs the actual **inference-time rollout**: same sampler, same step count, same stochastic AR cache the deployed model would see. This is the on-policy trajectory.

*Teacher.* Same model architecture, but conditioned on a **clean AR-consistent temporal cache** built from ground-truth long-video context. The teacher denoises the student's noisy latent at every diffusion step of the student's rollout, producing the correct denoising direction for that particular noisy latent.

**On-policy distillation loss.** Along the student's rollout, at every denoising step $s$ of every frame chunk $t$:
$$\mathcal{L}_{\text{OPSD-V}} = \sum_{t, s} \| f_{\text{student}}(z_{t,s}, \text{cache}^{\text{student}}_{<t}) - f_{\text{teacher}}(z_{t,s}, \text{cache}^{\text{clean}}_{<t}) \|^2$$
The student learns to produce the same denoising output the teacher would have produced *on the student's own trajectory*. Distillation is dense (per step, per frame) rather than only at trajectory endpoints.

**No inference change.** The sampler, step count, and cache structure at inference stay identical to the base model. All the change is in weights.

## Why it matters

- **Fixes drift without slowing inference.** The alternatives — teacher-forced training data, more diffusion steps, KV-rework — either don't fix drift or cost inference. OPSD-V is a training-time fix that leaves inference untouched.
- **Dense supervision beats endpoint supervision.** Prior long-video post-training compared final videos (too weak) or full-precision trajectories (too expensive). Per-step denoising correction is both dense and cheap.
- **Applied to Self-Forcing and LongLive**, OPSD-V wins **66.0% of user-preference judgments** (82.5% excluding ties) on long-rollout video quality.
- **Template transfers.** The pattern — student rolls out, teacher corrects on the student's own noisy trajectory — likely transfers to any few-step AR sequence model (audio, motion, tokens) where inference-time cache drift is the bottleneck.

## Gotchas & tricks

- **Teacher choice.** Teacher must be the *same base model* (or a large-step variant of it), not a stronger external model — the distillation is about closing the train/inference gap, not upgrading to a different model.
- **Cache asymmetry is the whole point.** If the teacher also uses the student's noisy cache, there is no supervisory signal. Keeping the teacher on a *clean* cache is what makes the correction target meaningful.
- **On-policy vs off-policy.** Rollouts must be sampled from the current student. Freezing rollouts halfway through training reintroduces off-policy skew.
- **Compute cost.** Running the teacher on every denoising step doubles training compute per rollout. Sample fewer rollouts per iteration or sub-sample denoising steps to control cost.
- **Long-tail failure preservation.** Rare drift modes (e.g. object teleportation after 30 seconds) may never appear in short training rollouts. Include long-context rollouts in the mixture.

## Sources

- Paper: *OPSD-V: On-Policy Self-Distillation for Post-Training Few-Step Autoregressive Video Generators* — Liu, Wang, Gao, He, Ma, Wan, Zhang, Wei, Chen (Meituan, HKUST, CityU HK), 2026 — arXiv:2607.08766.
- Base models cited: Self-Forcing, LongLive (few-step AR video diffusion).
