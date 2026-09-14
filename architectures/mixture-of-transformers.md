# Mixture-of-Transformers (MoT)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A modality-routed cousin of MoE where the "experts" are entire transformer sub-stacks specialized per modality (vision, text, action) sharing one attention substrate. Enables a single **encoder-free + VAE-free** backbone to understand, reason over, and generate images without a CLIP tower or a diffusion decoder. Used at 8B in SenseNova-U1.5.

**Prereqs:** [_moe](_moe.md), [transformer-block](transformer-block.md)
**Related:** [deepseek-moe](deepseek-moe.md), [_moe](_moe.md)

---

## What it is

MoE routes tokens to a subset of experts inside FFN blocks. MoT routes tokens to modality-specialized transformer sub-stacks based on the token's *modality tag*, not on a learned router. Vision tokens run through vision-expert blocks, text tokens through text-expert blocks, action tokens through action-expert blocks — but the attention operation itself is shared, so cross-modal information mixes for free through self-attention.

The design collapses two previously separate towers into one backbone: no external vision encoder (CLIP/ViT), no external VAE decoder. Everything is a sequence of modality-tagged tokens in one transformer.

## How it works

- **Token tagging.** Every input token carries a modality tag. Image patches → vision tag; text pieces → text tag; (optional) actions/audio → own tags.
- **Per-modality FFN + LN + projections.** Each modality has its own FFN weights, its own layer norms, and its own input/output projections. Choosing which set to apply is a hard function of the tag — no learned gate, no auxiliary balance loss.
- **Shared attention.** A single Q/K/V attention operation mixes tokens across modalities. This is where cross-modal binding happens (image ↔ caption ↔ action).
- **Unified sequence-to-sequence training.** Understanding, editing, and generation are all cast as autoregressive sequences over modality-tagged tokens. One loss, one curriculum.
- **No frozen encoder / decoder.** Image tokens are produced by a lightweight image tokenizer trained jointly (see [image-tokenizer-evaluation](../multimodal/image-tokenizer-evaluation.md)); pixels are decoded back through the same backbone rather than a VAE.

## Why it matters

Every open unified multimodal model to date has kept either a frozen vision encoder or a diffusion/VAE decoder to bridge modalities. MoT is the first credible open demonstration at 8B that neither is required — the same backbone perceives *and* generates. Compared to MoE, routing by modality tag eliminates load-balance losses, auxiliary router training, and gate-collapse pathologies: the assignment is data-defined.

## Gotchas & tricks

- **Per-modality LayerNorm matters more than per-modality FFN.** Sharing LayerNorm across modalities collapses the modality-specific statistics; ablations show LN is where the biggest wins live.
- **Shared attention is the whole point.** Splitting attention per modality reduces MoT to a hard-routed multi-tower system — worse than a joint transformer.
- **Tokenizer choice matters more than for VAE-based models** — the image tokenizer is now the pixel bottleneck; picking on FID misleads (see [image-tokenizer-evaluation](../multimodal/image-tokenizer-evaluation.md)).
- **Training curriculum stages modalities in.** Text-only warmup → text+vision understanding → generation → editing works better than joint from step 0.
- **No MoE-style capacity factor.** Because routing is deterministic, expert utilization is set by data composition — mix your modalities carefully.

## Sources

- Paper: *SenseNova-U1.5: Towards Native Unified Visual Intelligence* — SenseTime, 2026 — https://arxiv.org/abs/2609.11929
- Related: *Mixture-of-Transformers* line of work in prior multimodal MoT papers (see paper's references).
