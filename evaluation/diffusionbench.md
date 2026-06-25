# DiffusionBench
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Dual-evaluation framework for diffusion transformers (DiTs): a method must be evaluated on *both* class-conditional ImageNet generation *and* text-to-image generation. The paper introduces NanoGen, a unified training framework that supports both setups with a 12-line config change, then trains 21 latent diffusion models and shows Pearson correlations between −0.38 and −0.58 across metrics — a method that improves ImageNet FID often hurts T2I metrics.

**Prereqs:** [README.md](README.md)
**Related:** [../multimodal/README.md](../multimodal/README.md) · [counterfactual-t2i.md](counterfactual-t2i.md)

---

## What it is

DiT research has converged on class-conditional ImageNet generation as the canonical benchmark, mostly because T2I training is perceived as too expensive for ablations. DiffusionBench challenges that assumption with NanoGen — a lean training framework where T2I costs roughly what ImageNet costs — and uses NanoGen to run a head-to-head comparison across 21 trained latent diffusion models.

## How it works

Two pieces:

**NanoGen (training framework).** Unified pipeline supporting RAE, VAE, pixel-space, and MeanFlow diffusion variants under both ImageNet and T2I setups. A method change requires a single ~12-line config edit; training compute for T2I is comparable to ImageNet at matched model size.

**DiffusionBench (evaluation protocol).** Train each candidate method under both ImageNet and T2I in NanoGen, report both sets of metrics. The recommendation: papers should publish *both* numbers; ImageNet-only rankings are no longer informative.

The empirical finding that motivates the protocol: across the 21 models trained, the Pearson correlation between ImageNet FID and three T2I metrics ranges from −0.38 to −0.58. Negative — so on average, methods that improve ImageNet FID *reduce* T2I quality. The community's ImageNet-FID-driven progress is not just under-informative; it's actively misleading for T2I.

## Why it matters

- Forces the DiT community to confront a benchmark over-fit that has been suspected anecdotally and now has empirical receipts.
- NanoGen as a tool lowers the per-paper cost of doing the right thing (publishing T2I numbers alongside ImageNet).
- The negative correlation is a substantive claim about the structure of the design space — different sub-spaces of methods are good at different objectives.

## Gotchas & tricks

- Negative correlation is across the specific 21 models in this study; not a universal law. Methods that *intentionally* target T2I will of course show different patterns.
- Three T2I metrics were reported; choice of metrics matters and partly explains the magnitude of the correlation.
- NanoGen's lean framework abstracts only the methods it supports — methods outside the supported families need framework extension.
- The benchmark says "report both" rather than "rank by a single composite" — leaves the weighting question to downstream users.

## Sources

- Paper: *On Holistic Evaluation of Diffusion Transformers with a Unified Training Framework Bridging ImageNet and Text-to-Image* — Leng, Singh, Liang, Smith, Bell, Saha, Yuan, Zheng — ANU / Canva Research, 2026 — [arXiv:2606.24888](https://arxiv.org/abs/2606.24888).
