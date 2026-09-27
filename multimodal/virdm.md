# ViRDM — Representation Distribution Matching for Few-Step Video Diffusion
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A few-step distillation method for **causal (autoregressive) video diffusion** that drops the DMD setup (large teacher + online critic) and instead matches student and reference distributions directly in a frozen encoder's **representation space**. Three tricks — windowed matching, time-adjusted moments, and a temporal-consistency regulariser — make it work for video. Reaches VBench 84.87 with only 20 generator updates.

**Prereqs:** [README.md](README.md)
**Related:** [../post-training/_post-training.md](../post-training/_post-training.md)

---

## What it is

The dominant few-step diffusion distillation recipe is **Distribution Matching Distillation (DMD)**: run a fast student, run the full diffusion **teacher**, use an **online critic** to estimate their distributional discrepancy from diffusion scores, and pull the student toward the teacher. This is expensive — you need teacher-scale forward passes and a trainable critic — and awkward for video, where memory dominates.

ViRDM keeps the "distribution matching" goal but changes the matching *space*: match **feature-space statistics** (moments in a frozen encoder's embedding space) rather than score functions. That eliminates both the online critic and the need to run the teacher during distillation.

## How it works

Setup:

- Fast **student** $g_\phi$: a few-step causal video diffusion generator.
- Frozen **encoder** $\Phi$ shared with the teacher's training regime (e.g. a video CLIP/DINO trunk).
- A dataset of reference videos representing the target distribution.

Objective:

$$\mathcal{L} = \sum_k \| \mu_k(\Phi(g_\phi(z))) - \mu_k(\Phi(x_{\text{ref}})) \|_2^2 + \lambda_{\text{temp}} \mathcal{L}_{\text{temp}}$$

where $\mu_k$ are the first-through-$k$-th moments in feature space and $\mathcal{L}_{\text{temp}}$ is a temporal-consistency regulariser (see below). Three video-specific tricks:

1. **Windowed moment matching.** Compute moments over sliding windows of frames rather than the full clip, so memory doesn't blow up on long videos.
2. **Time-adjusted moment estimator.** Video training distributions have non-stationary statistics across frame index; ViRDM reweights moment estimates by frame position to correct for that.
3. **Temporal-consistency term.** A short-range regulariser penalising abrupt feature-space jumps between adjacent frames — constrains motion where the moment-matching loss is otherwise blind to per-step dynamics.

## Why it matters

- **No teacher, no critic.** Distillation collapses to a small student + a frozen encoder + a reference set — dramatically cheaper than DMD, especially at video scale.
- **20-update distillation.** ViRDM reaches VBench 84.87 with only 20 generator updates. That's fast enough that the method can be re-run after every base-model update rather than being a one-off distillation.
- **Portable idea.** The "distribution matching in a frozen representation space" recipe likely transfers to audio and 3D few-step generators; ViRDM is a case study.

## Gotchas & tricks

- **Encoder choice is load-bearing.** Match the encoder to the target distribution: a CLIP-style encoder rewards semantic fidelity; a video-dynamics encoder rewards motion realism. Wrong encoder → wrong distribution.
- **Higher moments are unstable.** Third and fourth moments have high variance at typical batch sizes; the paper uses first + second only, with the temporal term compensating for the lost fine structure.
- **Causal-only.** ViRDM is written for causal (autoregressive) video diffusion. For bidirectional video diffusion the temporal term needs a different formulation.
- **VBench ≠ user preference.** The 84.87 headline number is on VBench, which correlates but doesn't equate to human preference. Pair with a small preference eval before shipping.

## Sources

- Paper: *ViRDM: Taming Representation Distribution Matching for Few-Step Causal Video Generation* — Meng, Ge, Huang, Zhou, Jiang (Northeastern / Adobe), 2026, [arXiv 2609.28923](https://arxiv.org/abs/2609.28923) — introduces ViRDM and reports VBench 84.87 with 20 generator updates.
- Related: *Distribution Matching Distillation* — Yin et al., 2023 — the DMD baseline ViRDM replaces.
