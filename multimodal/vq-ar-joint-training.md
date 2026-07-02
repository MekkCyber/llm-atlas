# End-to-End Joint Training of VQ Tokenizer and AR Generator (GEAR)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Visual autoregressive (AR) generation typically freezes a pre-trained VQ tokenizer, then trains an AR generator on its discrete indices. This leaves the tokenizer *blind* to what the generator can predict easily. GEAR (Lin et al., 2026) trains both **end-to-end** via a **dual read-out** of the codebook assignment: a hard one-hot branch trains the AR with next-token prediction, and a differentiable soft branch carries a representation-alignment loss back to the tokenizer *only*, sidestepping VQ non-differentiability without a straight-through-estimator collapse.

**Prereqs:** [../fundamentals/_tokenization](../fundamentals/_tokenization.md), [README](README.md)
**Related:** [../fundamentals/bpe](../fundamentals/bpe.md)

---

## What it is

Two-stage recipes dominate visual AR generation (LlamaGen, Emu3, Chameleon):

1. Train a **VQ tokenizer** (encoder → codebook → decoder) with a reconstruction objective. Freeze it.
2. Train an **AR generator** on the discrete indices produced by the frozen tokenizer.

The decoupling has a known bottleneck: the tokenizer optimizes reconstruction, unaware of which discrete distributions the AR would find easy to model. If the AR consistently struggles on some codebook regime, there's no signal back to the tokenizer to change.

**GEAR closes the loop end-to-end.** The catch is that VQ index lookup is non-differentiable — a straight-through-estimator (STE) here empirically collapses the tokenizer. GEAR's dual read-out solves this.

## How it works

### Dual read-out of codebook assignment

For each spatial location, the encoder produces continuous feature $z$; the codebook is $\{e_1, ..., e_K\}$. Two parallel branches:

- **Hard branch.** Quantize $z$ to its nearest codebook entry $e_{i^\star}$; feed the discrete index $i^\star$ into the AR. Loss: standard AR next-token prediction. This branch trains the AR on *actual* discrete tokens.
- **Soft branch.** Compute a differentiable soft assignment $\tilde{e} = \sum_k \text{softmax}(-\|z - e_k\|^2 / \tau)_k \cdot e_k$. Feed $\tilde{e}$ through a *representation alignment loss* (e.g., DINOv2 alignment). This branch's gradient flows *only* back to the tokenizer (encoder + codebook), never into the AR.

### Which branch does what

- The hard branch trains the AR to predict discrete indices. Standard cross-entropy; no non-differentiability issue because the AR just needs the index as input.
- The soft branch *carries the tokenizer's alignment burden*. Because gradients don't reach the AR through this branch, the AR is never affected by the soft path — it always trains on hard tokens.

### Feature-shift finding

Empirically, training with the dual read-out shifts alignment across the two networks:

- The tokenizer's own features become *less* DINOv2-like.
- The AR's features become *more* DINOv2-like.

This is the **opposite** of diffusion-side recipes (REPA-style) that make the *latent* itself semantic. GEAR shifts the alignment burden from tokenizer to AR — the AR steers the tokenizer toward an index distribution *it* can predict more easily.

## Why it matters

- **Up to 10× convergence speedup** on ImageNet gFID over a strong LlamaGen-REPA baseline.
- **Better spatial features.** GEAR-trained AR models produce markedly better patch-level and spatially-coherent features — helpful for downstream editing and conditioning.
- **Generalizes across quantizers.** Works with VQVAE, LFQ, IBQ, and extends from class-conditional to text-to-image.
- **Composes with the broader AR-over-VQ family.** Any visual (or audio, video) system that uses "discrete tokens + AR generator" can slot in the dual read-out.

## Gotchas & tricks

- **The two branches must share the codebook and encoder** — not just conceptually, but literally the same parameters. Otherwise you have two separate tokenizers and the trick evaporates.
- **Soft branch temperature $\tau$ matters.** Very low $\tau$ recovers STE and collapses; very high $\tau$ produces a near-uniform soft assignment and washes out the alignment signal. Values around 1.0 are typical.
- **Stop-grad on the AR side.** Ensure the soft-branch loss's gradient is stopped at the AR boundary. Bugs here send AR-shaped noise into the tokenizer.
- **Alignment target choice.** DINOv2 is the paper's default; other self-supervised targets (MAE, DINO v3) may work but are untested. Whatever you choose, be consistent across pre-training runs.

## Sources

- Paper: *GEAR: Guided End-to-End AutoRegression for Image Synthesis* — Lin, Liu, Lin, Chen, Ge et al., 2026 — Peking University / Tencent Hunyuan.
- Related: LlamaGen, LlamaGen-REPA, VQ-Diffusion, MaskGIT — the AR-over-VQ family GEAR targets.
