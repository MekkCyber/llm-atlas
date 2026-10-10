# U-Space
*Depth — a low-dimensional subspace of the residual stream that captures when and why uncertainty arises in a language model.*

**TL;DR:** U-Space is a handful-of-dimensions subspace built from **doubt / certainty anchor prompts**. Project a model's hidden states onto it to get (a) a per-token **U-Lens** map showing where uncertainty spikes mid-generation, and (b) a scalar **U-score** that beats sampling-based baselines on reasoning benchmarks under length-controlled evaluation and transfers across models better than supervised probes. No labels, no repeated sampling, no training. Introduced in 2026.

**Prereqs:** [../fundamentals/attention.md](../fundamentals/attention.md), [README.md](README.md)
**Related:** [../post-training/rlvr.md](../post-training/rlvr.md) · [../safety/cot-monitoring.md](../safety/cot-monitoring.md)

---

## What it is

An activation-space method for uncertainty estimation at inference time. The construction is anchor-based: a small set of "I'm sure…" / "I'm not sure…" style prompts drives the model's residual stream in opposite directions; the span of those directions defines U-Space. Everything after that is projection.

Unlike supervised probes, U-Space needs no correctness labels. Unlike sampling-based estimators (semantic entropy, variance across samples), it needs no multiple generations. One forward pass yields both a token-level diagnostic and a scalar score.

## How it works

**Construction.** Pick $k$ pairs of anchor prompts whose only semantic difference is doubt vs. certainty. For each pair, record the final-layer (or a chosen layer) residual activations on a target token. The **mean of difference vectors** across pairs, together with its orthogonal complement under a small SVD, defines U-Space — effectively a few dimensions carrying most of the doubt↔certainty variance.

**U-Lens (per-token).** At inference, project every token's residual onto U-Space. The scalar coordinate along the "doubt" axis is the U-Lens signal — plot it along the generation and uncertainty spikes become visible.

**U-score (scalar).** Pool the per-token signals (mean / max over a span) to summarise uncertainty for an answer. Threshold to abstain, re-sample, or route to a stronger model.

## Why it matters

- **Sampling-free.** Uncertainty estimators that require $N$ samples per query multiply inference cost by $N$. U-Space is one forward pass.
- **Label-free.** No ground-truth answers required for construction; anchors are hand-written and reused across tasks.
- **Transfers across models.** Supervised probes often overfit to a specific model's residual geometry; U-Space, built from semantic anchors, moves across model families with less degradation.
- **Diagnostic, not just scalar.** The U-Lens makes long-reasoning traces debuggable — you can see which step a chain-of-thought lost confidence.

## Gotchas & tricks

- Anchor quality matters more than quantity; a well-chosen handful beats a scraped corpus.
- Choose the probing layer carefully — middle layers often outperform the final layer for uncertainty signals, since the final layer collapses onto logits.
- Length control is non-trivial: longer generations drift along U-Space even when the model is confident. Normalise by span length or compare at matched lengths.
- U-Space is a **relative** signal: it tells you doubt grew, not an absolute probability of being wrong. Calibration against a benchmark is needed before using it as an abstention threshold.

## Sources

- Paper: *U-Space: Uncovering When and Why Uncertainty Arises in Language Models* — Loose, Herzog et al., TU Darmstadt / UCL / Mila / MBZUAI, 2026 — [arXiv:2610.09087](https://arxiv.org/abs/2610.09087).
