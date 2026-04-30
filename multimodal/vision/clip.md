# CLIP (Contrastive Language-Image Pre-training)

*Depth — contrastive training on 400M image-text pairs; the vision encoder the VLM era is built on.*

**TL;DR:** Train a vision encoder + text encoder jointly by **contrastive learning**: for each batch of N image-text pairs, compute an N×N cosine similarity matrix, apply a symmetric InfoNCE loss that pulls matched pairs together and pushes unmatched pairs apart. After training, the vision encoder produces **image embeddings pre-aligned with text** — enabling **zero-shot classification** via text prompts (*"a photo of a {class}"*). Radford et al. (OpenAI, 2021). Trained on **400M pairs scraped from the web** ("WIT"), batch size **32,768**. Best model **ViT-L/14@336px**: 76.2% ImageNet zero-shot. The foundation for LLaVA, Flamingo, and nearly every VLM.

**Prereqs:** [vit](vit.md), [attention](../../fundamentals/attention.md)
**Related:** [siglip](siglip.md) · [metaclip](metaclip.md) · [cross-attention-adapter](cross-attention-adapter.md) · [llava](llava.md)

---

## What it is

A self-supervised training objective: given a batch of (image, text) pairs, learn encoders that map matched pairs to similar points in a shared embedding space and unmatched pairs to dissimilar points.

Radford et al., *Learning Transferable Visual Models From Natural Language Supervision*, ICML 2021, arXiv 2103.00020.

Instead of labeling images (expensive, constrained to a fixed taxonomy), **use natural language supervision from web image-alt-text pairs**. This trades label quality (noisy captions) for scale (400M pairs) and generality (any text-expressible concept becomes a classifier).

---

## How it works

### The objective (Sec. 2.3, Figure 3)

Given a batch of N (image, text) pairs:
- Encode images: `I_i = image_encoder(I) ∈ ℝ^d`, L2-normalized.
- Encode texts: `T_j = text_encoder(T) ∈ ℝ^d`, L2-normalized.
- Compute N×N similarity matrix with a temperature:
  ```
  logits_{ij} = I_i · T_j · exp(t)
  ```
  where `t = log(1/τ)` is a learnable scalar, clipped to prevent `exp(t) > 100`.
- The correct pairing is the diagonal (i, i). Symmetric cross-entropy:
  ```
  loss_i2t = cross_entropy(logits, labels=[0,1,...,N-1], axis=0)  # image→text
  loss_t2i = cross_entropy(logits, labels=[0,1,...,N-1], axis=1)  # text→image
  loss = (loss_i2t + loss_t2i) / 2
  ```

### Pseudocode (Figure 3)

```python
# Per-batch
I_e = l2_normalize(image_encoder(I), axis=1)      # [N, d]
T_e = l2_normalize(text_encoder(T),  axis=1)      # [N, d]

logits = (I_e @ T_e.T) * exp(t)                    # [N, N]
labels = arange(N)

loss_i = cross_entropy(logits, labels, axis=0)    # image → text
loss_t = cross_entropy(logits, labels, axis=1)    # text → image
loss   = (loss_i + loss_t) / 2
```

The softmax-contrastive-over-batch is a form of **InfoNCE** (Oord 2018). With a learnable temperature τ (or `t = log 1/τ`) that the model directly optimizes.

### The encoders

- **Image encoder**: ResNet variants (RN50, RN101, RN50x4, RN50x16, RN50x64 — EfficientNet-style scaling) OR **Vision Transformer variants (ViT-B/32, ViT-B/16, ViT-L/14, ViT-L/14@336px)**.
- **Text encoder**: Transformer decoder-style, **12 layers, width 512, 8 heads, ~63M params**. BPE vocab **49,152**. Max sequence length **76** tokens (+ `[SOS]`/`[EOS]`).
- Text representation: the `[EOS]` token's top-layer activation, **linearly projected** to the shared embedding space.
- Image representation: pooled vision-encoder output (attention-pooled for ResNet variants, `[CLS]`-token for ViT), also **linearly projected** to the shared space.
- Both embeddings **L2-normalized** before the cosine-similarity dot product.

### Dataset: WIT (Sec. 2.2)

- **400M (image, text) pairs** from the public web ("WebImageText").
- Collection: **~500,000 queries** (all English Wikipedia n-grams occurring ≥100 times + WordNet synsets + Wikipedia article titles).
- **Up to 20,000 pairs per query** for approximate class balance.
- Total word count comparable to WebText (used to train GPT-2).

### Training details (Sec. 2.5)

- **Batch size 32,768**. **32 epochs** over WIT.
- AdamW, decoupled weight decay on non-gain/bias weights; cosine LR schedule.
- **Learnable temperature τ initialized to ≈0.07** (so `exp(t) = 1/τ ≈ 14`); clipped so `exp(t) ≤ 100`.
- Mixed-precision training; half-precision Adam stats and text embeddings.
- Only data augmentation: random square crop from resized images.
- Compute: best model (RN50x64) trained 18 days on 592 V100s. ViT-L/14: 12 days on 256 V100s.

### Model variants (Table 1)

- ResNets: RN50, RN101, RN50x4, RN50x16, RN50x64.
- ViTs: **ViT-B/32, ViT-B/16, ViT-L/14, ViT-L/14@336px**.

`@336px` = fine-tuned for one additional epoch at 336×336 resolution, which slightly improves results.

### Zero-shot classification (Sec. 3.1)

Given K classes and their names:
- Encode each class via a prompt template, e.g., `"A photo of a {class}."` → K text embeddings.
- Encode the image → 1 image embedding.
- Softmax over cosine similarities (scaled by τ):
  ```
  p(class | image) = softmax(τ · (image_emb · text_embs^T))
  ```

No gradient steps needed, no supervised data for the target task. The image encoder's trained "concept-alignment" with text is the classifier.

**Prompt engineering** helps substantially:
- `"A photo of a {class}."` beats raw `"{class}"` by ~1.3%.
- **Ensembling 80 prompt templates** (`"a photo of a {c}"`, `"a cropped photo of a {c}"`, etc.) beats single template by ~3.5%.

### Results

Headline: **ViT-L/14@336px zero-shot ImageNet: 76.2%** (top-1). Ties supervised ResNet-50's accuracy **without ever training on ImageNet labels**.

Across 27 benchmarks, CLIP is broadly the best image encoder for transfer learning; features are near-universal.

---

## Why it matters

- **The canonical vision encoder for the VLM era.** CLIP's pretrained ViT-L/14 or ViT-B/16 is the default image encoder for LLaVA, Flamingo, BLIP-2, and countless follow-ups. Usually frozen during VLM training.
- **Language-aligned features.** Unlike ImageNet-supervised ViT (which learns "ImageNet-category-discriminating" features), CLIP learns features aligned with arbitrary text. Lets the LLM use the image features as "token-like" objects.
- **Zero-shot classification.** Any visual concept with a text description becomes a zero-shot classifier. Huge practical win.
- **Foundation for open-world vision.** Open-vocabulary detection (GroundingDINO, OWL-ViT), open-vocabulary segmentation (SAM-CLIP), and more all build on CLIP.
- **Demonstrates that captions are useful labels.** The paper's philosophical point: supervised labels are a narrow slice of the signal in an image; natural-language captions are a richer, more general signal.

---

## Gotchas & tricks

- **Text encoder is small and weak by design.** The text encoder's job is to produce good text *embeddings*, not to be a language model. 63M params, max 76 tokens — enough for prompts, not for paragraphs.
- **Embedding space is 512- or 768-dim typically.** After L2 normalization, similarities live on the unit sphere.
- **Temperature τ is load-bearing.** Too high: softmax spreads too uniformly, gradient vanishes. Too low: only one similarity dominates, no information from other pairs. Learnable τ finds the right setting automatically.
- **Batch size matters.** 32,768 pairs → 32,768² = 1B similarity comparisons per batch, with ~32K negatives per positive. Contrastive learning benefits from many negatives; batch size below ~8K gives measurably worse results (see SigLIP's analysis).
- **Data quality matters more than quantity.** CLIP's 400M is noisy; MetaCLIP (Xu 2024) showed a cleaner recipe produces better results at the same scale.
- **Hard-negative mining doesn't help much.** Random in-batch negatives are near-sufficient; more sophisticated mining gives minor gains.
- **Softmax vs sigmoid loss.** CLIP's softmax requires comparing against all in-batch negatives. SigLIP (Zhai 2023) uses per-pair sigmoid loss → linear in batch size, enables even larger batches, sometimes better quality.
- **Frozen at deployment.** For VLM training, the CLIP vision encoder is usually frozen (or lightly fine-tuned) — you want to preserve its language-aligned features. Fine-tuning too much breaks them.
- **Multilingual CLIP.** The original text encoder is English-only. Multilingual CLIP (mCLIP, XLM-CLIP) handles 100+ languages but at slightly lower English quality.
- **Failure modes.** CLIP struggles with fine-grained visual detail (bird species, car models), text rendering in images, counting, and compositional relations ("a cat on top of a dog" vs "a dog on top of a cat"). These are well-documented weaknesses.
- **Data leakage.** The "zero-shot ImageNet" claim has a caveat: WIT likely contains images captioned with ImageNet class names. Not truly zero-shot in the strict sense.
- **Fine-tuning protocols.** For downstream tasks, linear probing (linear layer on frozen features) captures most of CLIP's value; full fine-tuning can overfit on small datasets.

---

## Sources

- Paper: *Learning Transferable Visual Models From Natural Language Supervision* — Radford et al., OpenAI, ICML 2021, arXiv 2103.00020.
- Paper: *Representation Learning with Contrastive Predictive Coding (InfoNCE)* — Oord et al., 2018, arXiv 1807.03748 — the theoretical basis of CLIP's loss.
- Paper: *Demystifying CLIP Data (MetaCLIP)* — Xu et al., ICLR 2024 — reproducible data recipe, see [metaclip](metaclip.md).
- Paper: *Sigmoid Loss for Language-Image Pre-Training (SigLIP)* — Zhai et al., ICCV 2023, see [siglip](siglip.md).
- Paper: *An Image is Worth 16x16 Words (ViT)* — Dosovitskiy et al., 2020 — CLIP's ViT backbone.
