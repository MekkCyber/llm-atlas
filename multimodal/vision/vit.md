# Vision Transformer (ViT)

*Depth — treat image patches as tokens, feed to a Transformer, classify.*

**TL;DR:** Split a 224×224 image into 16×16 patches (or 14×14), flatten each to a 768-d vector, add a learnable `[CLS]` token and learned 1D positional embeddings, feed the resulting 197-token sequence to a standard Transformer encoder, classify via the `[CLS]` token's final-layer representation. No convolutions, no inductive biases beyond the patch split. Needs much more data than CNNs (JFT-300M or ImageNet-21k) to be competitive. ViT-H/14 at 88.55% ImageNet top-1 was a watershed — demonstrated "Transformer scales in vision too." The foundation for CLIP, MetaCLIP, SigLIP, and every modern vision encoder.

**Prereqs:** [attention](../../fundamentals/attention.md), [transformer-block](../../architectures/transformer-block.md)
**Related:** [clip](clip.md) · [siglip](siglip.md) · [llava](llava.md)

---

## What it is

An architecture that adapts the Transformer to images by **tokenizing the image into patches**. A "visual BERT" with images as input.

Dosovitskiy et al., *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*, ICLR 2021, arXiv 2010.11929.

Before ViT, computer vision was dominated by CNNs (ResNet, EfficientNet). The ViT paper's claim: if you have enough data, the Transformer is as good or better than CNNs for images, with fewer inductive biases. The key result: when pretrained on JFT-300M (Google's internal 303M-image dataset) and fine-tuned on ImageNet, ViT-H/14 hits **88.55% ImageNet top-1** (Table 2).

---

## How it works

### Patch embedding (Sec. 3.1, Eq. 1)

Input image `x ∈ ℝ^(H×W×C)` (for 224×224 RGB: `x ∈ ℝ^(224×224×3)`):

1. Split into **N patches** of size `P × P`: `N = HW / P² = 224² / 16² = 196` for P=16.
2. Flatten each patch: `x_p^i ∈ ℝ^(P²·C) = ℝ^(16·16·3) = ℝ^768`.
3. **Linear projection** E: `ℝ^(P²·C) → ℝ^D` (where D is the Transformer hidden size). Gives `N` patch embeddings.
4. **Prepend `[CLS]` token**: `x_class ∈ ℝ^D` (learnable). Total sequence length = N + 1.
5. **Add learned positional embeddings** `E_pos ∈ ℝ^((N+1)×D)`.

```python
class PatchEmbedding(nn.Module):
    def __init__(self, image_size=224, patch_size=16, in_chans=3, embed_dim=768):
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        # (equivalent to flatten + linear, but conv2d is cleaner)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        num_patches = (image_size // patch_size) ** 2  # 196 for 224/16
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

    def forward(self, x):  # x: [B, C, H, W]
        x = self.proj(x)                         # [B, D, 14, 14]
        x = x.flatten(2).transpose(1, 2)          # [B, 196, D]
        cls = self.cls_token.expand(x.size(0), -1, -1)  # [B, 1, D]
        x = torch.cat([cls, x], dim=1)            # [B, 197, D]
        x = x + self.pos_embed                    # [B, 197, D]
        return x
```

So z₀ (paper Eq. 1):

```
z₀ = [x_class; x_p¹·E; x_p²·E; …; x_p^N·E] + E_pos
```

### The encoder (Sec. 3.1, Eq. 2–3)

Standard pre-norm Transformer blocks:

```
z'_ℓ = MSA(LN(z_{ℓ-1})) + z_{ℓ-1}     (ℓ = 1 … L)
z_ℓ  = MLP(LN(z'_ℓ)) + z'_ℓ           (ℓ = 1 … L)
```

- **MSA**: multi-head self-attention.
- **MLP**: two-layer FFN with GELU.
- **LN**: LayerNorm (pre-norm).

### Classification head

```
y = LN(z_L⁰)            (take [CLS]'s final representation, LayerNorm-normalize)
logits = MLP(y)          (pretraining: 1-hidden-layer MLP with tanh)
         or linear(y)    (fine-tuning: single linear to target classes)
```

### Variants (Table 1)

| Model | Layers | D | MLP dim | Heads | Params |
|---|---|---|---|---|---|
| ViT-Base (B) | 12 | 768 | 3072 | 12 | 86M |
| ViT-Large (L) | 24 | 1024 | 4096 | 16 | 307M |
| ViT-Huge (H) | 32 | 1280 | 5120 | 16 | 632M |

Naming convention: **ViT-L/16** = Large with P=16 patches. Common ViT variants:
- ViT-B/32 (smaller seq length, less compute).
- ViT-B/16 (standard).
- ViT-L/16 (larger model).
- ViT-L/14 (more patches → more compute, finer spatial detail).
- ViT-H/14 (the largest, used by CLIP's best variant).

Sequence lengths at 224×224:
- P=32: N=49, total 50 tokens.
- P=16: N=196, total 197 tokens.
- P=14: N=256, total 257 tokens.

### Why the `[CLS]` token

Same purpose as BERT's `[CLS]`: a learnable token that aggregates global image information via attention. The final-layer representation of `[CLS]` is used for classification. Alternative: average-pool over patch tokens (used in some follow-ups like DeiT). Both work; ViT's original paper uses `[CLS]`.

### Position embeddings

**Learned 1D** in the original paper. The patches are ordered raster-scan (row-major). Higher-dim 2D position embeddings (splitting into row + column) gave minimal improvement (Appendix D.4), so the simpler 1D version became standard.

**Resolution changes at fine-tuning**: when fine-tuning at higher resolution (e.g., pretrain at 224, fine-tune at 384), the number of patches changes (196 → 576). The learned positional embeddings are **2D-interpolated** to the new grid — the paper found this works well.

### Training recipe (Table 3, Appendix B.1)

- Adam optimizer, β₁=0.9, β₂=0.999, weight decay 0.1.
- Batch size 4096.
- Linear LR warmup (10k steps), linear decay.
- Dropout 0 during JFT pretraining.
- Fine-tuning: SGD with momentum, batch 512, gradient clip 1.0, cosine schedule.

**Pretraining datasets** (in order of size):
- ImageNet-1K: 1.3M images, 1K classes. Not enough for ViT to beat CNNs.
- ImageNet-21K: ~14M images, 21K classes. Borderline.
- **JFT-300M**: 303M images, 18K classes. Where ViT starts to win.

The scaling conclusion: **ViT scales better than CNNs with more data and compute**. With ImageNet alone, ResNets win; with JFT, ViT wins. This is the "data-hungry" ViT result (Fig. 3).

---

## Why it matters

- **Transformer in vision.** Before ViT, CNNs dominated vision. ViT showed Transformers can match them at scale, opening the door to shared architectures across vision, language, audio, and multimodal.
- **Foundation for CLIP / SigLIP / DINO / SAM.** Every major 2022+ vision encoder builds on ViT. The `[CLS]` token (or average-pooled patches) gives a global image embedding that CLIP-style contrastive training can use.
- **Variable-resolution inference.** Because ViT is a Transformer, you can feed any-length patch sequence at inference. 2D interpolation of position embeddings handles different image sizes gracefully.
- **Composable with LLMs.** The patch-tokens-as-sequence abstraction plugs directly into LLM input pipelines. LLaVA, Flamingo, and BLIP-2 all consume ViT patch outputs.
- **Benchmarks.** ViT-H/14 on JFT-300M → 88.55% ImageNet — the strongest CNN results at the time were ~88%. Parity or slight edge.

---

## Gotchas & tricks

- **Needs a lot of data.** Pretraining on ImageNet-1K alone gives worse results than a comparable-size ResNet. Minimum ImageNet-21K, really JFT-300M. DeiT (Touvron 2020) found data augmentation + teacher distillation gets ViT competitive with just ImageNet — still harder than CNNs from scratch.
- **Patch size matters for spatial detail.** P=16 is the standard. P=14 increases sequence length 16/14²·N ≈ 1.3× but captures finer features; used by CLIP's ViT-L/14 and ViT-H/14.
- **Attention is quadratic in N.** At 224² with P=16: N=196, attention cost O(N²)=38K. At 512²: N=1024, attention O(N²)=1M. Fine-tuning at higher resolution is expensive.
- **Pretraining objective matters.** ViT's original is supervised classification on JFT labels. Later: masked autoencoding (MAE, He 2021), contrastive (CLIP), DINO (self-distillation), SAM (segmentation). The architecture is the same; the objective is what the encoder learns.
- **Inductive biases matter less at scale.** The "CNNs have locality / translation-equivariance baked in; ViT doesn't" critique matters at ImageNet scale but washes out at JFT-300M. At scale, learned patterns > baked-in priors.
- **Positional encoding interpolation for higher resolution.** 2D bilinear/bicubic works for moderate resolution changes. For very large changes (224 → 1024), more sophisticated interpolation or re-training helps.
- **No causal mask needed.** Vision is non-causal — every patch attends to every other patch. Full dense attention by default.
- **Register tokens.** Follow-ups (Darcet 2023) add a few extra learnable "register" tokens that act like `[CLS]` but for spatial reasoning. Slightly improves results.
- **`[CLS]` vs avg-pool for downstream.** For classification, both work; for CLIP-style retrieval, avg-pool of patch tokens is sometimes preferred. Task-dependent.
- **Data augmentation hurts ViT less than CNNs.** ViT responds well to RandAugment / Mixup / CutMix; for small-data pretraining these are critical.

---

## Sources

- Paper: *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale* — Dosovitskiy et al., ICLR 2021, arXiv 2010.11929.
- Paper: *Training data-efficient image transformers & distillation through attention (DeiT)* — Touvron et al., ICML 2021, arXiv 2012.12877 — data-efficient ViT via distillation.
- Paper: *Masked Autoencoders Are Scalable Vision Learners (MAE)* — He et al., CVPR 2022, arXiv 2111.06377 — self-supervised ViT pretraining.
- Paper: *Emerging Properties in Self-Supervised Vision Transformers (DINO)* — Caron et al., ICCV 2021, arXiv 2104.14294.
- Paper: *Vision Transformers Need Registers* — Darcet et al., ICLR 2024, arXiv 2309.16588.
