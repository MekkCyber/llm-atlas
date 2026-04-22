# Normalization

*Taxonomy — the layers that rescale activations inside a transformer block, and where you place them.*

**TL;DR:** Every modern transformer has normalization layers sprinkled through it, doing the same job at every location: rescale activations so that downstream layers see inputs with bounded magnitude. Modern LLMs have converged on **RMSNorm** (the layer) in a **pre-norm or reordered-norm placement** (the position), with **QK-norm** as a stability-adding extension inside attention. The choice of *which* normalization matters less than choosing *that* you're going to normalize and where.

**Related taxonomies:** [_training-stability](../pre-training/_training-stability.md)
**Depth files covered here:** [qk-norm](qk-norm.md) · [reordered-norm](reordered-norm.md)

---

## The problem

Activations in a deep network can drift in magnitude across layers. Without normalization:

- Gradients vanish or explode during backprop.
- Attention softmax saturates as Q/K magnitudes grow.
- Output logits drift in magnitude, making the softmax pathologically sharp.
- Training becomes fragile at depth and at long horizons — loss spikes, NaNs.

Normalization is the general answer: at specific points in the network, rescale activations to bounded magnitude with a learned scale (and sometimes shift). The question is *what* to rescale and *where* to put the rescaling.

## The shared pattern

Every normalization layer follows the same template:

```
y = (x - shift) / scale           # normalize
y = y * γ + β                     # optional learned affine
```

Variants differ on three axes:

1. **What `shift` and `scale` are computed from** — mean+variance (LayerNorm), RMS only (RMSNorm), batch statistics (BatchNorm), per-group (GroupNorm).
2. **Over which axis they're computed** — feature dim (Layer/RMSNorm), batch dim (BatchNorm), spatial (GroupNorm).
3. **Whether the affine params (γ, β) are present, learned, or fixed** — RMSNorm drops β, some variants drop γ too.

A separate, orthogonal design choice is **placement** inside the transformer block: before the sub-layer (pre-norm), after (post-norm), on the sub-layer output before residual merge (reordered-norm). This matters as much as the layer choice itself.

## Variants

### Normalization layers

| Technique | Normalizes over | Centered? | Learned affine | Used in |
| --- | --- | --- | --- | --- |
| LayerNorm | Feature dim, per-token | Yes (mean + var) | γ and β | Original Transformer, GPT-2, BERT |
| **RMSNorm** | Feature dim, per-token | No (RMS only) | γ only | **Modern LLMs (LLaMA, Mistral, Qwen, DeepSeek, OLMo)** |
| [QK-norm](qk-norm.md) | Feature dim of Q and K, per-head per-token | No (RMS) | γ only | OLMo 2, Gemma 2, Scaling ViT 22B |
| BatchNorm | Batch + spatial, per-feature | Yes | γ and β | CNNs; poor fit for NLP (variable-length sequences) |
| GroupNorm / InstanceNorm | Per-group or per-sample | Yes | γ and β | Vision, diffusion; rare in LLMs |

### Placement variants

| Placement | Formula | Properties | Used in |
| --- | --- | --- | --- |
| Post-norm | `Norm(x + Sublayer(x))` | Norm in the residual path; hard to train deep | Original Transformer |
| **Pre-norm** | `x + Sublayer(Norm(x))` | Clean residual gradient; residual magnitude drifts at depth | **Most modern LLMs (LLaMA, Mistral, GPT-NeoX)** |
| [Reordered-norm](reordered-norm.md) | `x + Norm(Sublayer(x))` | Clean residual *and* bounded sub-layer contribution | OLMo 2 |
| Sandwich norm | `x + Norm(Sublayer(Norm(x)))` | Two norms per sub-layer; extra safety, extra compute | CogView, some specialized runs |

## How to choose

**For a modern LLM from scratch:**

- **Layer choice: RMSNorm.** Cheaper than LayerNorm (no mean subtraction, one learned parameter set instead of two), benchmarks are indistinguishable. No reason to pick LayerNorm unless you're restoring a specific architecture for compatibility.
- **Placement: pre-norm or [reordered-norm](reordered-norm.md).** Pre-norm is the broadly safe default. Reordered-norm is modestly better for long / deep training runs where you care about the residual-stream drift issue.
- **Extension: add [QK-norm](qk-norm.md) inside attention.** Two extra RMSNorms per block (on Q and K, per-head), catches a specific instability class that pre-norm alone misses. Essentially free; modern stability-conscious runs include it.

Stacking these three — RMSNorm + reordered placement + QK-norm — is the OLMo 2 recipe, and closes most of the "loss diverged at step 400k" failure class.

**Post-norm** is historical for LLMs. If you're reading a pre-2020 paper it's probably post-norm; anything modern is pre-norm or reordered-norm. Don't resurrect post-norm at scale.

**BatchNorm, GroupNorm, InstanceNorm** — skip for LLMs. BatchNorm in particular is a bad fit because LLM batches have variable sequence lengths and the statistics across the padding confuse it.

## Adjacent but distinct

- **[Z-loss](../fundamentals/z-loss.md)** — bounds the final output logits' magnitude via loss regularization, not via a normalization layer. Stability technique, not a norm.
- **Weight normalization** — reparametrize weights as direction × magnitude. Mostly historical; not what people mean by "normalization" in modern LLMs.
- **Gradient clipping** — bounds the gradient update norm, not the activation magnitude. Related-but-different stability technique.

## Sources

- Paper: *Layer Normalization* — Ba, Kiros, Hinton, 2016 — the original LayerNorm.
- Paper: *Root Mean Square Layer Normalization* — Zhang & Sennrich, 2019 — introduces RMSNorm, shows near-equivalent quality at lower cost.
- Paper: *On Layer Normalization in the Transformer Architecture* — Xiong et al., 2020 — theoretical analysis of pre-norm vs. post-norm.
- Paper: *Scaling Vision Transformers to 22 Billion Parameters* — Dehghani et al., 2023 — introduces QK-norm explicitly as a depth-stability fix.
- Paper: *2 OLMo 2 Furious* — AI2, 2024 — reordered-norm + QK-norm + RMSNorm ablations.
- Paper: *Batch Normalization* — Ioffe & Szegedy, 2015 — for the adjacent technique.
