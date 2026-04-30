# Multimodal Fusion Patterns

*Taxonomy — how non-text modalities (vision, audio) get plugged into language models.*

**TL;DR:** Three dominant patterns for bolting vision/audio onto an LLM: **cross-attention adapters** (Flamingo, Llama 3 vision — frozen LLM, learnable adapter), **direct projection** (LLaVA — projector + optional LLM fine-tune), and **Q-Former / learned-query bottleneck** (BLIP-2 — compress modality features to a fixed set of query tokens). Each has a different trainable-parameter footprint, visual-reasoning ceiling, and integration pattern. Choice is driven by: do you want to preserve the LLM's text quality exactly (cross-attention), or maximize visual reasoning at the cost of fine-tuning (direct projection)?

**Related taxonomies:** *(none yet)*
**Depth files covered here:** [cross-attention-adapter](vision/cross-attention-adapter.md) · [llava](vision/llava.md) · [q-former](vision/q-former.md)

---

## The problem

You have a pretrained LLM (Chinchilla, Llama, Vicuna) and a pretrained vision encoder (CLIP ViT, SigLIP SO400M). You want a model that can take images as input and produce text output. Three failure modes to dodge:

1. **Damaging the LLM's text capability.** Jointly fine-tuning the LLM on visual instruction data can cause catastrophic forgetting of language reasoning.
2. **Insufficient visual grounding.** The LLM doesn't "really" attend to the image; it pattern-matches from the prompt.
3. **Excessive parameter cost.** Training a full-model VLM from scratch is expensive; finding the smallest adapter that works is valuable.

The three main patterns below trade these off differently.

---

## The shared pattern

All multimodal fusion does the same thing: **convert modality features to a representation that the LLM can consume**. They differ in:

- **What's frozen**: LLM, encoder, both, neither.
- **How fusion happens**: cross-attention (attention from LLM to modality features), direct insertion (modality features become LLM tokens), bottleneck (modality → fixed queries → LLM).
- **When fusion happens**: in every LLM layer (cross-attention), once at the input (direct projection / Q-Former).

---

## Variants

| Pattern | LLM | Encoder | Fusion | Trainable params | Visual ceiling | Key reference |
|---|---|---|---|---|---|---|
| **[Cross-attention adapter](vision/cross-attention-adapter.md)** | Frozen | Frozen | Gated cross-attention in LLM layers | Medium (~10% of LLM) | Medium-high | Flamingo, Llama 3 vision |
| **[Direct projection](vision/llava.md)** (LLaVA) | **Fine-tuned** | Frozen | MLP projector → LLM tokens | High (full LLM + small projector) | High | LLaVA-1.5 |
| **[Q-Former](vision/q-former.md)** | Frozen | Frozen | 32-query bottleneck → soft visual prompts | Low (~100-200M) | Medium | BLIP-2 |
| **Early fusion** | Co-trained | Co-trained | Modality tokens interleaved from scratch | Everything | Highest ceiling | Gemini (reported), PaLI-X |

### Cross-attention adapter pattern

- **Insertion**: cross-attention layers inserted between frozen LLM blocks (every 4 layers typical).
- **Gating**: `tanh(α)` initialized to 0 — adapter is no-op at start, preserves LLM initially.
- **Trainable**: cross-attention + a small Perceiver Resampler (64 latents, fixed-size).
- **Wins**: preserves text quality perfectly; scales to 405B LLMs without catastrophic forgetting risk.
- **Loses**: visual reasoning capped by the frozen LLM's ability to process cross-attention signals.
- **Canonical**: Flamingo (70B frozen Chinchilla + 10B trainable adapter), Llama 3 vision (405B frozen + ~100B cross-attn).

### Direct projection pattern

- **Fusion**: vision features → (linear or MLP) projector → concatenated into LLM's token stream.
- **Trainable**: projector + LLM (stage-2).
- **Wins**: highest visual-reasoning ceiling; simplest architecture; fits on a single node.
- **Loses**: LLM text quality at risk from fine-tune; need to mix text-only data to prevent forgetting.
- **Canonical**: LLaVA-1.5, MiniGPT-4, InternVL, Qwen-VL.

### Q-Former pattern

- **Fusion**: vision features → ~188M Q-Former with 32 learnable queries → 1-layer projection → soft prompts prepended to LLM.
- **Trainable**: Q-Former + 1-layer projection (~108M total).
- **Wins**: lowest trainable-param cost; LLM text quality perfectly preserved.
- **Loses**: 32-token bottleneck limits visual detail; two-stage training is compute-heavy for the small final model.
- **Canonical**: BLIP-2, InstructBLIP.

### Early-fusion pattern

- **Fusion**: modality tokens mixed with text tokens from the start; full model trained end-to-end from scratch.
- **Trainable**: everything.
- **Wins**: highest possible capability ceiling; no architectural seams.
- **Loses**: extremely expensive; requires full pretraining compute budget; no leverage from pretrained LLM.
- **Canonical**: Gemini (Google claims early fusion; details sparse), PaLI-X.

---

## How to choose

Decision tree for a new VLM project:

```
Do you have a pretrained LLM you want to preserve text quality on?
│
├── Yes → Do you care more about cost or ceiling?
│         │
│         ├── Cost matters more → Q-Former (~100M trainable, frozen both sides)
│         │
│         └── Ceiling matters more → Cross-attention adapter (frozen LLM, trainable adapter)
│                                     Flamingo / Llama 3 vision
│
└── No, willing to fine-tune LLM → Direct projection (LLaVA)
                                     Highest ceiling, simplest, full fine-tune

Willing to train from scratch? → Early fusion (Gemini-class budget)
```

### When to prefer each

- **Flamingo / cross-attention**: you have a 70B+ LLM you don't want to damage; you can afford 1-10B trainable adapter; you need perfect text quality.
- **LLaVA / direct projection**: you're on a budget; you're okay with a 7-13B scale; you want the highest visual reasoning per dollar.
- **BLIP-2 / Q-Former**: you want maximum parameter efficiency; visual tasks are reasonably simple (captioning, VQA, not detailed reasoning).
- **Early fusion**: you have Google-scale compute.

---

## Adjacent but distinct

- **Tool use with a vision OCR/API.** Not fusion — the LLM calls a vision tool, gets back text. Simpler; skips all the architectural questions. Used by e.g., Claude with Computer Use.
- **Speech integration.** Similar pattern but for audio. Llama 3 uses direct projection for speech (speech encoder → adapter → direct token insertion), which is unlike its vision recipe (cross-attention). See [audio/](audio/).
- **Visual tokenization (VQ-GAN, Chameleon).** Convert image to a sequence of discrete tokens from a learned codebook, treat identically to text. Enables unified autoregressive modeling. Used by Chameleon (Meta 2024), Parti.
- **Continuous visual generation.** Instead of text output, output pixels (DALL-E 2, Imagen). Opposite direction — LLM-driven vision generation vs vision-conditioned LLM.

---

## Sources

- Paper: *Flamingo* — Alayrac et al., 2022 — cross-attention adapter canonical reference.
- Paper: *BLIP-2* — Li et al., 2023 — Q-Former canonical reference.
- Paper: *LLaVA* — Liu et al., 2023, and *LLaVA-1.5* — Liu et al., 2023 — direct-projection canonical reference.
- Paper: *PaLI-X* — Chen et al., 2023 — early-fusion variant.
- Paper: *Chameleon* — Meta, 2024, arXiv 2405.09818 — visual-tokenization variant.
- Paper: *The Llama 3 Herd of Models* — Meta, 2024, Sec. 7 (vision) + Sec. 8 (speech) — uses cross-attention for vision, direct token insertion for speech.
