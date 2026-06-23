# Diffusion Multimodal LLMs

*Depth — multimodal LLMs whose language head is a mask diffusion model rather than autoregressive, enabling parallel decoding for dense vision-language tasks.*

**TL;DR:** Almost every production multimodal LLM (MLLM) generates text autoregressively. A **diffusion MLLM** swaps the AR text decoder for a [mask diffusion language model](../fundamentals/mask-diffusion-lm.md): visual tokens fuse into the model the usual way, but text generation runs as iterative parallel denoising over a masked sequence. The promise is *parallel decoding* for vision-language tasks — especially valuable for *dense* tasks where you'd otherwise issue N separate AR forward passes. Demonstrated as PerceptionDLM (Sun et al., 2026) for parallel region perception.

**Prereqs:** [../fundamentals/mask-diffusion-lm.md](../fundamentals/mask-diffusion-lm.md)
**Related:** [../architectures/multi-head-attention.md](../architectures/multi-head-attention.md), [vla](vla.md)

---

## What it is

A multimodal LLM family with three components:

1. **A vision encoder** (CLIP / SigLIP / native pixel) that produces a sequence of visual tokens.
2. **A projection / fusion layer** that maps visual tokens into the LM's embedding space (linear projection, Q-former, or cross-attention).
3. **A mask-diffusion text decoder** that generates the response by iterative parallel denoising — *not* autoregressively.

The vision side mirrors a standard MLLM; the language side is the [MDM](../fundamentals/mask-diffusion-lm.md) family. The interesting design space is how the visual tokens condition the denoising — typically by prepending them to the masked text sequence and using bidirectional attention so every text position can attend to every visual token at every denoising step.

## How it works

Forward pass (one denoising step):

```
Image I → vision_encoder → V (visual tokens)
Prompt template: [V_tokens] [text_tokens with some masked]
Bidirectional attention over the full sequence
Predict masked positions in parallel → commit highest-confidence
Repeat T times until clean text remains
```

For **parallel dense tasks** (e.g. captioning multiple image regions, answering multiple questions about one image): instead of issuing N independent forward passes, structure the prompt with N separate "answer slots" and use [structured attention masks](../architectures/multi-head-attention.md) that isolate each slot's prompt from the others while letting them all attend to the visual tokens. A single denoising trajectory then fills every slot simultaneously.

## Why it matters

The pitch for diffusion MLLMs over AR MLLMs has two parts:

- **Parallel decoding on the language side.** Diffusion MLLMs decode many tokens per denoising step; AR MLLMs decode one. On standard single-response tasks the wall-clock advantage depends on T (number of denoising steps) vs. the AR sequence length, and isn't always a win. But on dense tasks the advantage is structural: N regions in one trajectory, not N separate AR runs.
- **Sequence-level parallelism.** Multiple independent answer slots in one prompt + structured attention masking = one forward pass for N outputs. This is the regime where PerceptionDLM beats AR MLLMs cleanly.

Current state of the art (mid-2026): diffusion MLLMs lag AR MLLMs on standard VQA / captioning benchmarks at parameter parity, but match or exceed on parallel-perception tasks. The race is far from over.

## Gotchas & tricks

- **Number of denoising steps T is the dominant cost knob.** Too few → quality drops; too many → no wall-clock advantage over AR. Schedule design matters.
- **Visual-token conditioning is sensitive.** Bidirectional attention works well for natural-language denoising but can over-attend to the (small, fixed) visual prefix, suppressing diversity. Some implementations cross-attend to visual tokens separately.
- **Parallel slots need careful prompt construction.** A single shared visual prefix + per-region prompt slots + an attention mask that isolates slots is the working pattern; sharing too much between slots breaks per-slot specificity.
- **Pretraining recipes are still maturing.** A diffusion MLLM benefits from pretraining the MDM backbone on text-only data first, then continuing pretraining with image-text data — closer to the BERT-then-multimodal recipe than the AR-MLLM standard.

## Sources

- Paper: *PerceptionDLM: Parallel Region Perception with Multimodal Diffusion Language Models* — Sun et al., 2026 — https://arxiv.org/abs/2606.19534
- Foundational: *Show-o: One Single Transformer to Unify Multimodal Understanding and Generation* — Xie et al., 2024 — early unified diffusion MLLM.
- Foundational: *MMaDA: Multimodal Large Diffusion Language Models* — Yang et al., 2025 — prior diffusion-MLLM baseline.
