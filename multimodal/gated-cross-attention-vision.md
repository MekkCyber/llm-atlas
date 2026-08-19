# Gated Cross-Attention for Vision-Language Models
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Most modern VLMs concatenate visual tokens into the language sequence, so text generation stops while the KV cache absorbs the new visual context. **Gated cross-attention** puts vision on a *separate* attention pathway into the decoder: the language decoder attends to text as usual, and a gated cross-attention block adds vision on top. Streaming frames don't invalidate the text KV cache, so the model can *see while it speaks* — the design MOSS-VL (Fudan, 2026) uses to make real-time interaction a first-class capability.

**Prereqs:** [../multimodal/README.md](README.md), [../architectures/multi-head-attention.md](../architectures/multi-head-attention.md)
**Related:** [streaming-vlm.md](streaming-vlm.md) · [../case-studies/moss-vl.md](../case-studies/moss-vl.md)

---

## What it is

Two mainstream VLM designs today:

1. **Prefix-token VLMs (LLaVA-style)** — visual tokens from an encoder are projected and concatenated in front of the text tokens. Self-attention over the joined sequence handles both. New frames mean invalidating the text KV cache and restarting from the joined context.
2. **Gated cross-attention VLMs (Flamingo, MOSS-VL)** — the text decoder is unchanged and processes text with its own self-attention; **inserted cross-attention layers** (each gated by a learned scalar/vector initialized to zero) additionally attend to vision features. Vision is *added* to the residual stream rather than *concatenated* into the sequence.

Under the second design, the text KV cache is untouched by a new frame — you can stream vision into the cross-attention pathway while text generation continues.

## How it works

Insert a cross-attention block every $k$ decoder layers:

$$
h \leftarrow h + \tanh(\gamma) \cdot \text{CrossAttn}(Q{=}h, K{=}V{=}\text{VisionFeatures})
$$

Design notes:

- **$\gamma$ is a learned gate**, initialized to 0. At the start of training the cross-attention block is a no-op — the model is exactly the pre-trained LLM. As $\gamma$ moves off 0, vision starts contributing. This is what keeps the language capability intact through multimodal adaptation.
- Cross-attention queries come from the language hidden state; keys/values come from a pooled or projected vision-encoder output.
- The text self-attention pathway is untouched — its KV cache is a text-only cache.
- Streaming: new frames update the vision KV/features feeding cross-attention; text generation reads them on the next token without invalidation.

## Why it matters

- **Native "see while speak" behavior.** Text generation isn't gated by frame arrival, and frame arrival isn't gated by text generation. Real-time voice+vision agents (screen assistants, live-video helpers) need exactly this.
- **Preserves the base LLM.** Zero-init gating means the pre-trained LLM's next-token distribution starts unchanged; multimodal training can lift capability without regressing text.
- **Cleaner KV-cache economics.** The text cache stays small and text-only; the vision-side cache can be shorter-lived (frames age out). Two caches with separate lifetimes are easier to serve than one merged cache with mixed modality.

## Gotchas & tricks

- **Cross-attention block placement matters.** Too few blocks → vision under-integrated; too many → language quality erodes and compute goes up. Common defaults: every 4–8 decoder layers.
- **Gate init must be zero (or close).** A random init breaks the no-op-at-start property and forces the LLM to unlearn drift before it can integrate vision.
- **Frame-rate tuning.** With streaming vision, the model receives many frames per generated token. Under-throttling the cross-attention key/value updates wastes compute; over-throttling drops information.
- **Not the same as adapter/LoRA on the vision path.** Adapters modify existing weights; gated cross-attention adds *new* weights on a separate pathway. Both can be combined.
- **Vision encoder choice still dominates.** Regardless of integration style, a weak vision encoder caps the ceiling.

## Sources

- Paper: *MOSS-VL Technical Report* — Wang, Tan, Zhou et al. — arXiv:2608.15045 — 2026 (Fudan University / Shanghai Innovation Institute).
- Origin of the pattern: *Flamingo* — Alayrac et al., DeepMind, 2022 — introduced perceiver-resampled visual features + gated cross-attention into a frozen LLM.
