# Vision & Vision-Language Models

*Depth files for the vision encoder families and the integration patterns that bolt vision onto LLMs.*

---

## Reading order

1. **[vit](vit.md)** — the Vision Transformer. Patches-as-tokens; the architectural foundation for every modern vision encoder.
2. **[clip](clip.md)** — contrastive image-text pretraining. The default way to produce a vision encoder with pre-aligned text semantics.
3. **[metaclip](metaclip.md)** — CLIP with a published reproducible data recipe.
4. **[siglip](siglip.md)** — CLIP with sigmoid (per-pair) loss instead of softmax; enables larger batches and is the current open default.
5. **[cross-attention-adapter](cross-attention-adapter.md)** — the Flamingo pattern: frozen LLM + frozen vision encoder + gated cross-attention. Used by Llama 3's vision variant.
6. **[q-former](q-former.md)** — BLIP-2's lightweight bridge between vision encoder and LLM. The compact-query-vector alternative to cross-attention adapters.
7. **[llava](llava.md)** — the simplest production VLM recipe: vision encoder → projector → LLM, train with visual instruction-tuning. Most open VLMs follow this pattern.

## Related

- [_multimodal-fusion](_multimodal-fusion.md) — taxonomy of integration patterns.
- [audio/](../audio/) — analogous ladder on the speech side.
- [architectures/](../../architectures/) — LLM architectural primitives these vision stacks bolt onto.
