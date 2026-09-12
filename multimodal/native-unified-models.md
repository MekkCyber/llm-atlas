# Native Unified Multimodal Models
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A **native unified** multimodal model does understanding, reasoning, and generation of visual content in **one end-to-end network** — no external vision encoder (CLIP-family), no external VAE latent bridge for generation. The backbone consumes raw image patches and emits raw image patches (or a lightweight discrete image tokenizer's codes). SenseNova-U1.5 is the 8B canonical example: encoder-free + VAE-free, trained with **spatially-coherent patch reconstruction**, post-trained by **multi-expert on-policy distillation** across visual aesthetics, bilingual text rendering, infographics, and image editing.

**Prereqs:** [mixture-of-transformers](mixture-of-transformers.md)
**Related:** [image-tokenizers](README.md), [../case-studies/sensenova-u1-5](../case-studies/sensenova-u1-5.md)

---

## What it is

Prior unified multimodal models stitched three pretrained systems together: (1) a vision encoder (SigLIP, CLIP) for understanding, (2) an LLM backbone for reasoning, and (3) a diffusion model or VAE latent decoder for generation. Native unified drops (1) and (3): the backbone learns its *own* patch-level representation on the input side and produces its *own* patch-level output on the generation side. The whole model trains under one objective schedule.

Two definitional properties:
- **Encoder-free.** No CLIP/SigLIP tower. Image patches are embedded by a simple linear + norm and enter the transformer directly.
- **VAE-free.** No pretrained VAE. Generation targets are either raw patch pixels (via reconstruction objectives) or codes from a discrete image tokenizer trained *jointly* with the backbone, not a frozen VAE latent.

## How it works

**Input path.** Image → patches → linear embed → per-modality projection into the shared attention stream (typically via a Mixture-of-Transformers layout, one expert per modality).

**Understanding objective.** Standard next-token prediction over interleaved text+patch sequences. The model sees images and text jointly and answers questions in text.

**Generation objective.** The backbone emits patches (or discrete image codes) autoregressively. Native resolution can go high (SenseNova-U1.5 reports up to 4K) because there's no latent-space bottleneck.

**Spatially-coherent patch reconstruction.** A dense auxiliary loss on the visual interface: mask random patches during pretraining and reconstruct them from surrounding patches through the backbone. Provides pixel-level supervision without requiring a separate VAE.

**Post-training via multi-expert on-policy distillation.** Train specialized experts on-policy (visual aesthetics, bilingual glyph rendering, infographics, editing), then distill their behaviors back into the unified policy so one model retains all specialist capabilities.

## Why it matters

- **Collapses the stack.** Three trained systems → one trained system. Less duplicated compute, no interface drift between encoder-decoder mismatches.
- **Higher native resolution.** VAE bottlenecks capped generation at ~1K–2K; native pixel-space paths reach 4K in SenseNova-U1.5.
- **Bi-directional transfer.** Understanding representations reach into generation and vice versa within one loss surface — visual planning transfers into visual creation.
- **Fewer moving pieces at inference.** Serving one model instead of three pipelines.

## Gotchas & tricks

- **Patch reconstruction weight.** Too high and the model becomes a pixel encoder that fails to reason; too low and the visual interface degenerates. SenseNova-U1.5's ratio is not public.
- **Cold-start on generation.** Without a frozen VAE, the model has to *learn* pixel-space generation from scratch; early training benefits from a curriculum that starts with reconstruction then progressively opens the generation objective.
- **Native resolution ≠ 4K everywhere.** Training at 4K sparsely; most training runs at moderate resolutions with high-res only in curated batches.
- **Post-training expert consolidation is fragile.** Distilling multiple specialist experts into one model can collapse the weaker ones; interleave replay from each expert's training distribution.
- **Not a replacement for diffusion.** Native unified generation trades sample quality for architectural simplicity in some regimes; specialized diffusion models can still win on unconditional image fidelity.

## Sources

- Paper: *SenseNova-U1.5: Towards Native Unified Visual Intelligence* — SenseTime, 2026 — [arXiv:2609.11929](https://arxiv.org/abs/2609.11929)
- Related: earlier unified models with encoders/VAEs (Chameleon, Show-o, Janus, Emu3) — the ancestors this design abandons pieces of.
