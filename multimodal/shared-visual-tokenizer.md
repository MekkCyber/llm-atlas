# Shared Context-Visual Tokenizer (UniAR)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Use **one** discrete visual tokenizer for both multimodal *understanding* and *generation*, so a unified autoregressive multimodal model can read its own generated tokens without re-encoding. UniAR adds lookup-free bitwise quantization and parallel bitwise prediction to shrink the visual sequence length, achieving SOTA image generation and editing while staying competitive on understanding.

**Prereqs:** [../fundamentals/_tokenization.md](../fundamentals/_tokenization.md)
**Related:** (multimodal generation — no general page yet)

---

## What it is

"Unified" multimodal LLMs typically run two tokenizers:

- A **semantic encoder** (CLIP / SigLIP family) for understanding — outputs continuous features.
- A **VQ-VAE-style codebook** for generation — outputs discrete codes optimized for pixel reconstruction.

This duality is awkward: when the model generates an image and then reasons about it, the output codes must be re-encoded through the semantic encoder before they can be understood. Closed-loop reasoning over generated content (e.g., "draw this, then critique it") becomes brittle and expensive.

UniAR commits to a **single shared discrete tokenizer** for both directions. The model can directly interpret its own generated tokens without any re-encoding step.

---

## How it works

### The single tokenizer

Trained to simultaneously support two objectives:
- **Reconstruction**, so the codes carry pixel-faithful information for generation.
- **Multimodal alignment** (vision-language contrast / understanding loss), so the codes carry semantic information for understanding.

The output is a single discrete code per spatial position usable on both sides.

### Multi-level feature fusion

To support both objectives with one codebook, UniAR fuses *multiple levels* of the visual feature hierarchy before quantization. Early-layer features carry low-level/structural information needed for reconstruction; late-layer features carry semantic information needed for understanding. Both are pooled into the tokenizer input.

### Lookup-free bitwise quantization

Standard VQ quantizes into a finite codebook by nearest-neighbor lookup. UniAR uses **bitwise quantization** — each code is a $b$-bit string where each bit is decided independently by a learned threshold on a continuous feature. There is no explicit codebook table; the "codebook" is implicit (all $2^b$ bit strings).

Advantages:
- No codebook collapse (a common VQ failure mode).
- Larger effective vocabulary at fixed memory cost.
- Bits decompose independently for parallel prediction.

### Parallel bitwise prediction

Because the $b$ bits of each token are independent given the model state, the autoregressive head predicts all $b$ bits in parallel per spatial position. This shortens the effective sequence length seen by the AR loop by a factor of $b$ — important for image-scale spatial grids.

---

## Why it matters

- **Removes the two-tokenizer wart** that has limited closed-loop multimodal models.
- **Closed-loop reasoning over generated images** becomes natural: the model can think about what it just produced without an autoencoder round-trip.
- **SOTA image generation and editing benchmarks** while remaining competitive on understanding — the shared tokenizer doesn't trade off either side.
- **Bitwise quantization is independently useful.** It's a robust alternative to VQ-VAE codebooks that has applications beyond multimodal unification.

---

## Gotchas & tricks

- **Both objectives must train together.** Bolting reconstruction onto a frozen semantic encoder doesn't work; the tokenizer needs to be optimized jointly for both.
- **Bit count $b$ is a knob.** More bits per token = more capacity, longer parallel predictions, slower training. Paper sweep is implicit; treat $b$ as model-size-dependent.
- **Parallel bitwise prediction breaks naive ARsequence loss.** The training objective sums per-bit cross-entropy at each spatial position; conditioning across spatial positions stays AR.
- **Independence assumption on bits.** Bits are predicted independently given the model state. If the bits encode correlated structural information, prediction quality degrades. Spatial AR conditioning carries most of the correlation; the per-token bits are designed to be near-independent.

---

## Sources

- Paper: *Unified Multimodal Autoregressive Modeling with Shared Context-Visual Tokenizer is Key to Unification* — Wujian Peng et al. (Qwen-affiliated team), 2026 — [arXiv:2606.18249](https://arxiv.org/abs/2606.18249).
