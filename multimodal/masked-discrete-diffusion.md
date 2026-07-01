# Masked Discrete Diffusion (MDM) for Image Synthesis
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Masked discrete diffusion generates images by iteratively unmasking a grid of discrete visual tokens. It's the natural discrete counterpart of continuous latent diffusion, but historically has two structural weaknesses: **no self-correction** (unmasked tokens can't be revised) and **signal sparsity** when the codebook is large. NLD-Image fixes both — a **token-editing** mechanism that revises already-unmasked tokens during inference, and a **Grouped Cross-Entropy (GCE)** objective that spreads positive learning signal to embedding-space neighbours. Result: 0.90 GenEval, 86.9 DPG, 10.76 HPSv3.

**Prereqs:** [README.md](./README.md), [../fundamentals/attention.md](../fundamentals/attention.md)
**Related:** [../architectures/_moe.md](../architectures/_moe.md)

---

## What it is

Continuous diffusion iteratively refines a noisy latent — every step touches the entire latent, so mistakes can be revised. Masked discrete diffusion (MDM) instead operates on a grid of *discrete* visual tokens (from a VQ tokeniser): each step *unmasks* some tokens by predicting them from a masked-in-context; unmasked tokens are then treated as ground truth for later steps.

That difference has two consequences:
- **No self-correction:** an unmasked token can't be revised in later steps — a bad early commit propagates.
- **Signal sparsity with large codebooks:** larger vocab improves the VQ tokeniser's reconstruction fidelity but shrinks the per-token gradient (one right label out of tens of thousands).

## How it works

**Token editing (inference).**
- Standard MDM decoding: `mask → predict → commit → predict → commit`. Committed tokens are frozen.
- With token editing: at each decoding step, allow *already-committed* tokens to be re-predicted with some probability, then keep the higher-confidence version.
- Cheap to add at inference; no retraining required. Mirrors how a sculptor iteratively refines their work — the metaphor the paper uses.

**Grouped Cross-Entropy (GCE) — training.**
- Standard cross-entropy: exactly one positive label per masked position; gradient is nearly zero at large vocab.
- GCE: assign *positive* signal to the ground-truth token **and** to tokens neighbouring it in embedding space (nearest-k in the VQ codebook).
- Concretely, the objective becomes a soft cross-entropy over a small neighbourhood — denser per-token gradient, less variance.

**Custom fused GCE kernel.** GCE's neighbourhood computation is memory-heavy in a naive implementation at large vocab. The paper implements a fused CUDA operator that keeps VRAM usage tractable and speeds up training.

**Results.**
- **GenEval:** 0.90
- **DPG:** 86.9
- **HPSv3:** 10.76

Competitive with continuous-diffusion SOTA on the same benchmarks — MDM has closed the gap.

## Why it matters

- **Discrete substrate composes cleanly with autoregressive multimodal LMs.** If images are discrete tokens, the same transformer can do text and image generation over a shared vocabulary. MDM's revised viability makes any-to-any models cheaper.
- **Two general design principles.** *Editing after commit* generalises to any semi-autoregressive decoder; *neighbourhood-smoothed cross-entropy* is a lever for any large-vocabulary generative model.
- **Training efficiency.** GCE reduces the training-signal sparsity that limited MDM scaling; combined with the fused operator, training cost per quality point drops meaningfully.

## Gotchas & tricks

- **Token editing has diminishing returns past a few extra passes.** Additional refinement steps buy quality, but at inference cost — typically 1–2 revisions per token in practice.
- **GCE neighbourhood size is a knob.** Too small → still sparse; too large → over-smoothed, blurred outputs. Paper's default sits in the 5–10 nearest-neighbour range.
- **Editing requires confidence calibration.** Naively re-predicting can *degrade* good early commits if the model is under-confident on the second pass; the paper's rule prefers the higher-confidence prediction.
- **VQ tokeniser choice matters more than for continuous diffusion.** Large codebooks help GCE more than they help vanilla MDM — plan the tokeniser and objective together.

## Sources

- Paper: *Advancing Masked Discrete Diffusion for High-Resolution Image Synthesis* (NLD-Image / Nemotron-Labs-Diffusion-Image) — Heinrich, Ye, Fu, Grover, Kautz, Molchanov (NVIDIA / UCLA), 2026 — [arXiv:2606.29814](https://arxiv.org/abs/2606.29814).
