# Q-Former (BLIP-2)

*Depth — a lightweight bridge between frozen vision encoder and frozen LLM, using learnable queries.*

**TL;DR:** BLIP-2's contribution: a **~188M-param Transformer with 32 learnable query vectors** that cross-attend to a frozen vision encoder's outputs. Then a 1-layer projection turns these 32 queries into "soft visual prompts" prepended to the frozen LLM's text tokens. Two-stage training: (1) **representation learning** with image-text contrastive/matching/generation objectives, (2) **generative pre-training** against the frozen LLM. **Only the Q-Former + projection trained** — vision encoder and LLM stay frozen. Beats Flamingo-80B on VQAv2 zero-shot with ~54× fewer trainable params. The compact-query alternative to gated cross-attention.

**Prereqs:** [clip](clip.md), [vit](vit.md), [attention](../../fundamentals/attention.md)
**Related:** [cross-attention-adapter](cross-attention-adapter.md) · [llava](llava.md)

---

## What it is

Li et al., *BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models*, ICML 2023, arXiv 2301.12597.

An alternative to Flamingo's gated cross-attention: instead of inserting cross-attention throughout the LLM, **extract a small fixed number of visual tokens** via a learned bottleneck and **prepend them to the LLM's input sequence**.

- Flamingo: cross-attention layers inside the LLM, full vision features available to attend to.
- BLIP-2: one pass of "compress image → 32 tokens" upfront, LLM sees only the 32 tokens.

Much cheaper per forward pass; potentially less visual fidelity; simpler to deploy.

---

## How it works

### Q-Former architecture (Sec. 3.1, Figure 2)

Lightweight Transformer with two internal sub-modules:

1. **Image Transformer**: `32 learnable query embeddings (dim=768)` attend to each other (self-attn) and to frozen image features (cross-attn, inserted every other block). Self-attn initialized from BERTbase; cross-attn randomly initialized.

2. **Text Transformer**: same self-attention weights as the image transformer (shared), plus text token inputs for contrastive/matching/generation objectives.

Total Q-Former parameters: **~188M**.

Bottleneck: output is 32 × 768 = ~25K values. The raw vision encoder might produce `257 × 1024 ≈ 263K` values per image. **~10× compression**.

### Stage 1: Representation learning (Sec. 3.2)

Train the Q-Former with frozen vision encoder via three simultaneous objectives:

- **Image-Text Contrastive (ITC)**: unimodal self-attn mask (queries and text can't see each other). Cosine similarity between max-pooled queries and text `[CLS]` → InfoNCE with in-batch negatives.
- **Image-grounded Text Generation (ITG)**: multimodal causal self-attn mask. Generate text conditioned on queries. Uses `[DEC]` token instead of `[CLS]`.
- **Image-Text Matching (ITM)**: bidirectional self-attn mask. Binary classifier on each query's final representation, averaged.

The three objectives use **different attention masks** to implement the different tasks in the same Q-Former parameters — a neat multi-task trick.

Hard-negative mining for ITM (sample hard negatives via ITC similarities).

### Stage 2: Generative pre-training (Sec. 3.3)

Q-Former's output (32 × 768) → **1-layer linear projection** to the LLM's input-embedding dim → **prepended as 32 "soft visual prompts"** to the LLM's text token sequence.

```
LLM input = [soft_prompt_1, soft_prompt_2, ..., soft_prompt_32, text_token_1, text_token_2, ...]
```

- For **decoder-only LLMs** (OPT-2.7B, OPT-6.7B): standard next-token loss on the text continuation.
- For **encoder-decoder LLMs** (FlanT5-XL, FlanT5-XXL): prefix-LM split — encoder sees `[visual prefix; text prefix]`, decoder generates suffix.

**Only the Q-Former + 1-layer projection** train in stage 2. The LLM stays frozen.

### Training data (Sec. 3.4)

- **129M images**: COCO, Visual Genome, CC3M, CC12M, SBU, **115M from LAION-400M**.
- **CapFilt** (from BLIP-1): for noisy web images, generate 10 synthetic captions with BLIP-large, rank by CLIP similarity, keep top 2. Mix with originals.

### Training hyperparameters (Sec. 3.4)

- AdamW, β₁=0.9, β₂=0.98, weight decay 0.05.
- Cosine LR. Peak LR 1e-4, warmup 2K steps, min LR 5e-5 in stage 2.
- Stage 1: 250K steps, batch 2,320 (ViT-L) / 1,680 (ViT-g), 224×224, FP16.
- Stage 2: 80K steps, batch 1,920 (OPT) / 1,520 (FlanT5), FP16 (OPT) / BFloat16 (FlanT5).
- Compute: ViT-g + FlanT5-XXL fits on one 16×A100(40GB) machine; <6 days stage 1, <3 days stage 2.

### Parameter counts (Table 1)

- BLIP-2 ViT-L + OPT-2.7B: 3.1B total, **104M trainable**.
- BLIP-2 ViT-g + FlanT5-XXL: 12.1B total, **108M trainable**.
- Flamingo-80B: **~10B trainable** — ~54× more than BLIP-2.

Despite ~54× fewer trainable params, BLIP-2 beats Flamingo on VQAv2 zero-shot (Table 1).

---

## Why it matters

- **Extreme trainable-param efficiency.** 108M trainable vs Flamingo's 10B. Enables VLM training on a single server.
- **Introduces the "soft visual prompt" pattern.** 32 tokens-as-image has become a common abstraction. Variants include Perceiver Resampler in Flamingo (64 latents), LLaVA-1.5's direct-projection (no reduction).
- **Two-stage training is a reusable recipe.** Pretrain the bridge on vision-language-alignment tasks first; then generative pretraining against LLM. Several successors follow this template.
- **Multi-objective in one module.** Using different attention masks for different objectives in the same Q-Former is an efficient parameterization that later papers reuse.

---

## Gotchas & tricks

- **32 queries is a hard bottleneck.** Complex scenes with many objects may not fit into 32 tokens of representation. For detailed visual reasoning, LLaVA-style direct projection (every patch → every token) has more headroom.
- **Vision encoder frozen = encoder limits apply.** If the vision encoder can't distinguish dog breeds, neither can the Q-Former downstream.
- **LLM frozen = LLM limits apply.** Hallucination, reasoning failures inherited.
- **Stage 1 is expensive.** 250K steps across 129M images. Stage 2 is comparatively cheap.
- **Hard-negative mining for ITM.** BLIP-2 uses ITC similarity-based hard negatives. Required for ITM to work well — without them, easy negatives give vanishing gradients.
- **Output dim 768 tied to the Q-Former's width.** Doesn't scale naturally with LLM size — a 4096-dim LLM needs the 1-layer projection to bridge 768 → 4096.
- **Doesn't support interleaved images natively.** BLIP-2 assumes one image per sequence. For multi-image, you need to concatenate multiple Q-Former outputs — works but wasn't the design target.
- **No video handling built-in.** For video, you'd need a temporal aggregator; InstructBLIP and Video-LLaVA extend this.
- **Beaten by direct-projection methods at scale.** LLaVA-1.5 (projector is a 2-layer MLP, vision features go in as-is, no Q-Former bottleneck) outperforms BLIP-2 on many benchmarks with simpler machinery. The Q-Former bottleneck is a constraint, not always a feature.

---

## Sources

- Paper: *BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models* — Li, Li, Savarese, Hoi, Salesforce, ICML 2023, arXiv 2301.12597.
- Paper: *InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning* — Dai et al., 2023, arXiv 2305.06500 — Q-Former with instruction tuning.
- Repo: LAVIS — https://github.com/salesforce/LAVIS — reference implementation.
- Paper: *Perceiver IO* — Jaegle et al., 2022 — the learned-query-bottleneck family Q-Former belongs to.
