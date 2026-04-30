# Cross-Attention Adapter (Flamingo pattern)

*Depth — the integration pattern that bolts a frozen vision encoder onto a frozen LLM via learnable gated cross-attention layers.*

**TL;DR:** Freeze a pretrained LLM, freeze a pretrained vision encoder, insert **Perceiver Resampler** + **Gated Cross-Attention layers** in between, and train only the adapter (Resampler + cross-attn). Cross-attention layers are inserted between frozen LLM blocks; **`tanh` gating initialized to 0** makes the adapted model behave identically to the base LLM at step 0 — training stability guarantee. Introduced by Flamingo (Alayrac 2022); used by Llama 3's vision variant with cross-attention blocks every 4th LLM layer. For Llama 3 405B, the cross-attention layers alone are **~100B parameters**.

**Prereqs:** [attention](../../fundamentals/attention.md), [vit](vit.md), [clip](clip.md)
**Related:** [q-former](q-former.md) · [llava](llava.md)

---

## What it is

A strategy for creating a VLM by **grafting** vision onto a pretrained LLM without retraining the LLM. Core insight: preserve the LLM's pretrained knowledge; add *just enough* trainable machinery to let the model attend to image features when the input includes images.

Alayrac et al., *Flamingo: a Visual Language Model for Few-Shot Learning*, NeurIPS 2022, arXiv 2204.14198. The canonical reference. Llama 3 adopts the same pattern for its vision variant (Sec. 7).

Three key components:

1. **Frozen vision encoder** (typically CLIP ViT or similar): produces per-patch features.
2. **Perceiver Resampler**: reduces a variable number of patch features to a **fixed number of latent tokens** (64 in Flamingo, similar in Llama 3 vision). Bridges the variable image-feature shape to a fixed adapter-input shape.
3. **Gated Cross-Attention layers**: inserted between frozen LLM blocks; allow LLM tokens to attend to vision features.

Only the Perceiver Resampler + cross-attention layers are trained. The LLM weights stay frozen — the LLM's language capability is untouched.

---

## How it works

### The Perceiver Resampler

A small Transformer that takes a variable-size grid of vision features `X_f ∈ ℝ^(T × d)` (T = number of patches, variable across images/videos) and outputs a **fixed number of latent tokens** `L ∈ ℝ^(R × d)`. R is typically 64.

```python
class PerceiverResampler(nn.Module):
    def __init__(self, num_latents=64, dim=768, num_layers=6, num_heads=8):
        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        self.layers = nn.ModuleList([
            PerceiverBlock(dim, num_heads) for _ in range(num_layers)
        ])

    def forward(self, X_f):  # [B, T, d]
        latents = self.latents.unsqueeze(0).expand(X_f.size(0), -1, -1)  # [B, R, d]
        for layer in self.layers:
            # Cross-attend: queries = latents, keys/values = concat([X_f, latents])
            # (the Flamingo Resampler concatenates features and latents as KV)
            latents = layer(latents, torch.cat([X_f, latents], dim=1))
        return latents  # [B, R, d]
```

Design details:
- `R = 64` learnable query vectors.
- Architecturally close to Perceiver (Jaegle 2021) / DETR (Carion 2020) queries.
- Keys and values = **concatenation** of `[X_f; latents]`, per the Flamingo paper (not just X_f) — the paper reports this "doubles-K,V" variant works best.
- **Output: fixed-size representation** regardless of number of input patches (or frames, for video).

### The Gated Cross-Attention layer

Inserted between frozen LLM blocks. Architecture:

```
[frozen LLM block] → XAttn_gated → FFW_gated → [frozen LLM block]
```

The XAttn_gated block:

```python
def gated_xattn_dense(y, x_visual, alpha_xattn, alpha_dense):
    # y: LLM hidden state (sequence of text tokens)
    # x_visual: Perceiver Resampler output (fixed 64 latents)
    # alpha_xattn, alpha_dense: learnable scalars, INITIALIZED TO 0

    # Cross-attention: queries from LLM state, keys/values from vision
    y_xattn = cross_attention(q=y, kv=x_visual)
    y = y + tanh(alpha_xattn) * y_xattn   # gated residual

    # Feedforward
    y_ffn = FFN(y)
    y = y + tanh(alpha_dense) * y_ffn     # gated residual

    return y
```

The `tanh` gating is the critical piece:
- **At initialization**: `alpha_xattn = alpha_dense = 0` → `tanh(0) = 0` → **zero contribution**. The adapter is a no-op. The LLM sees its own hidden state unchanged, so the model behaves identically to the frozen LLM.
- **During training**: alphas grow away from 0. The adapter becomes active. The LLM gradually learns to attend to vision.

This is the stability guarantee. Without gating, inserting a random-init XAttn layer would disrupt the frozen LLM's carefully-learned representations; training would collapse the pretrained capability. With gating, you get a smooth handoff from "frozen LLM" to "vision-aware LLM."

### Insertion frequency

- **Flamingo**: every layer for small models; **every 4 layers for Flamingo-80B** (the 70B backbone).
- **Llama 3 405B vision variant (Sec. 7)**: **every 4 layers**. Cross-attention layers alone are ~100B parameters.

Sparser insertion reduces the adapter's parameter count and inference cost.

### Masking for interleaved images

Flamingo handles multi-image sequences: text interleaved with multiple images. The attention masking (Sec. 2.3) ensures each text token attends **only to the most recent preceding image's vision features** (not to all prior images). The LLM's self-attention propagates information about earlier images indirectly through residual streams.

This scheme, trained on sequences with ≤5 images, generalizes to **32-image sequences at inference** — enabling few-shot in-context visual learning.

---

## Training

### What's trainable

- **Frozen**: vision encoder (NFNet-F6 in Flamingo; CLIP-class encoder in Llama 3), LLM.
- **Trainable**: Perceiver Resampler + all Gated Cross-Attention blocks.

Trainable parameters per variant (Flamingo Table 2):
- Flamingo-3B: **~1.4B trainable** (on 1.4B frozen LM).
- Flamingo-9B: **~1.8B trainable** (on 7B frozen LM).
- Flamingo-80B: **~10B trainable** (on 70B frozen Chinchilla).
- Llama 3 405B vision: **~100B trainable** cross-attn parameters.

The cross-attention parameter count grows substantially with model size because each cross-attn layer has KV projections sized to the LLM's hidden dimension.

### Objective

Next-token prediction on interleaved image-text sequences. Same language modeling loss as the frozen LLM; gradients flow only through the trainable adapter.

### Data

Flamingo's training data:
- **M3W (MultiModal MassiveWeb)**: ~43M webpages with interleaved images + text → ~185M images, ~182 GB text.
- **ALIGN**: 1.8B noisy image-alt-text pairs.
- **LTIP** (Long Text Image Pairs): 312M high-quality image-text pairs.
- **VTP** (Video Text Pairs): 27M.

**Training sequences**: 256 tokens, ≤5 images per sequence. Grad-accumulation across datasets each step (Eq. 2).

### Frozen-LLM consequences

Because the LLM is frozen:
- **Preserves language capability perfectly** — no catastrophic forgetting.
- Text-only performance is **identical to the frozen LLM** (the adapter is a no-op for text-only inputs).
- **Cannot modify the LLM's failure modes** on text — if the LLM can't reason about Paris, adding vision doesn't help.

Contrast with LLaVA (which fine-tunes the LLM): higher ceiling for visual reasoning, but catastrophic forgetting risk.

---

## Why it matters

- **The canonical "plug vision into LLM" pattern.** Flamingo is the reference; every frozen-LLM VLM (including Llama 3 vision, Kosmos, IDEFICS) uses this.
- **Zero forgetting.** Text capabilities are perfectly preserved.
- **Composes with any LLM.** Swap the LLM (Chinchilla → Llama → Claude), retrain adapter only. No retraining the LLM.
- **The gated-init trick is broadly reusable.** Any time you're bolting new modules onto a frozen network, `tanh`-gated-to-zero initialization helps.
- **Scales elegantly.** Grafting onto a frozen 70B or 405B LLM works because you're only training the adapter — relatively small and fast.

---

## Gotchas & tricks

- **Gating is load-bearing.** Without `tanh(α)` gating at init, inserting cross-attention into a trained LLM tanks performance. Always use gated init for adapters.
- **Insertion frequency is a hyperparameter.** Every layer is too expensive at scale; every 4 layers is the Llama 3 / Flamingo-80B default. More layers = stronger visual grounding, higher cost.
- **Perceiver Resampler's latent count (R=64) is a bottleneck.** Too few latents → info loss; too many → expensive cross-attn. 32–128 is the typical range.
- **KV = [X_f; latents] concat (Flamingo) vs KV = X_f only.** The concat variant works better empirically. Not obvious why; Flamingo's ablations show it.
- **Cross-attention is expensive at inference.** 405B × ~20 vision layers × 64-token KV per image → measurable decode-time overhead. Serving systems need to cache vision features.
- **Doesn't "mix" vision into every layer.** Only specific layers have access. Some visual reasoning failures are because later LLM layers can't re-query vision.
- **Llama 3 unfreezes the image encoder.** Flamingo keeps the vision encoder frozen; Llama 3 (Sec. 7) unfreezes it during adapter training, claiming this improves text recognition in images.
- **Video adds a temporal aggregator.** Flamingo uses 1fps per video frame + learned temporal embeddings. Llama 3 vision has a Perceiver-Resampler-like temporal aggregator merging 32 consecutive frames into 1.
- **Multi-image interleaving mask is specific to Flamingo.** Not all VLMs adopt it; some just concatenate all images' features and let the LLM sort it out.
- **VLM fine-tuning post-adapter-training.** Some recipes later unfreeze more of the LLM for a small fine-tune. Trade-off between preservation and visual ceiling.

---

## Sources

- Paper: *Flamingo: a Visual Language Model for Few-Shot Learning* — Alayrac et al., DeepMind, NeurIPS 2022, arXiv 2204.14198.
- Paper: *Perceiver IO: A General Architecture for Structured Inputs & Outputs* — Jaegle et al., ICLR 2022, arXiv 2107.14795 — Perceiver architecture.
- Paper: *End-to-End Object Detection with Transformers (DETR)* — Carion et al., ECCV 2020, arXiv 2005.12872 — learned-query cross-attention precedent.
- Paper: *The Llama 3 Herd of Models* — Meta, 2024, arXiv 2407.21783, Sec. 7 — applies the Flamingo pattern to Llama 3 at 405B scale.
- Paper: *IDEFICS* — Hugging Face, 2023 — open reproduction of Flamingo.
