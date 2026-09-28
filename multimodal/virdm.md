# ViRDM — Representation Distribution Matching for Few-Step Video
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A **teacher- and critic-free** post-training recipe for few-step causal video diffusion. Replaces the three-network Distribution Matching Distillation (DMD) stack — teacher + online critic + generator — with a **generator-only** update that matches a **precomputed representation-distribution target**. Backed by three engineering tricks (stochastically-truncated clean-exit supervision, a lightweight VAE decoder, staged vector–Jacobian products) that make the recipe memory-feasible for multi-step causal video rollouts.

**Prereqs:** [README.md](README.md), [../post-training/_post-training.md](../post-training/_post-training.md)
**Related:** [../post-training/rejection-sampling.md](../post-training/rejection-sampling.md)

---

## What it is

Few-step causal (autoregressive) video generation is the streaming-friendly regime — low latency, but hard to train. Prior post-training relies on **Distribution Matching Distillation** (DMD): a big pretrained teacher supplies the "true" distribution, an online critic estimates the score, and the generator is nudged to match. Three networks in memory, one of them (the teacher) enormous.

ViRDM asks: can we drop the teacher and the critic and match a **precomputed** target instead? Adapting Representation Distribution Matching (RDM, previously used for one-step *image* generation) to *causal video* is the technical work.

## How it works

Three barriers the paper identifies and then addresses:

1. **Memory-intractable gradient path.** Multi-step causal rollouts backprop through many steps. Fix: **stochastically-truncated clean-exit supervision** — randomly cut off backprop at a clean intermediate state, so gradient depth is bounded.
2. **A video-specific optimization regime.** Video has different signal-to-noise structure than one-step image RDM. Fix: **staged vector–Jacobian products** — chunked gradient application that respects the causal ordering.
3. **Underconstrained temporal dynamics.** Representation matching alone doesn't pin frame-to-frame consistency. Fix: a **lightweight dynamics regularizer** on the generator's temporal features.

A lightweight VAE decoder reduces per-frame memory further, letting the whole thing fit in a small GPU budget.

## Why it matters

- 84.87 on VBench with **20 generator updates** on **16 A100 GPU-hours** — a fraction of the compute needed for DMD-style distillation.
- Killing the online critic and teacher changes the *economics* of few-step video: distillation is no longer a two-model exercise.
- The recipe extends to lower causal-sampling budgets and to 1/2/4-step bidirectional generation — a wider generality than the paper's headline setting suggests.

## Gotchas & tricks

- The target distribution has to be precomputed once, but the "right" reference (which frames, which features, how many) is a live design choice — not obvious a priori for a new video domain.
- Stochastic truncation trades gradient bias for memory; too-short truncations underfit long-horizon dynamics.
- The dynamics regularizer is small but non-optional — without it, temporal coherence degrades even though per-frame RDM matches.

## Sources

- Paper: *ViRDM: Taming Representation Distribution Matching for Few-Step Causal Video Generation* — Meng, Ge, Huang, Zhou, Jiang — 2026 — [arXiv:2609.28923](https://arxiv.org/abs/2609.28923).
