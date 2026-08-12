# Differential Attention
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Compute attention as the **difference between two softmaxes** on the same tokens — one "positive" attention map and one "noise-canceling" negative map — and use the difference as the final attention weights. The subtraction cancels attention noise on irrelevant tokens (a chronic pathology of vanilla softmax attention) and produces sharper, more selective attention distributions. Introduced 2024 as a plug-in replacement for standard attention; picked up in 2026-era frontier MoEs as one half of Grouped Differential Latent Attention.

**Prereqs:** [../fundamentals/attention.md](../fundamentals/attention.md), [multi-head-attention.md](multi-head-attention.md)
**Related:** [gdla.md](gdla.md), [mla.md](mla.md), [qk-norm.md](qk-norm.md)

---

## What it is

Standard softmax attention distributes probability mass over *all* keys — including irrelevant ones. Ablation studies show 30–50% of attention mass typically lands on tokens that don't matter for the output. This "attention noise" wastes model capacity, hurts long-context recall, and contributes to hallucination.

Differential attention modifies each attention head to compute **two attention maps** with independent parameters and returns their difference. The construction is analogous to noise-cancelling headphones: pass the same signal through two channels, one with the noise added coherently and one without, and subtract.

## How it works

Standard multi-head attention head:
```
head = softmax( (Q · K^T) / √d ) · V
```

Differential attention splits the head's Q and K into two halves and computes two attention maps:
```
Q₁, Q₂ = split(Q)         K₁, K₂ = split(K)
A₁ = softmax( (Q₁ · K₁^T) / √d )
A₂ = softmax( (Q₂ · K₂^T) / √d )
head = ( A₁ − λ · A₂ ) · V
```

`λ` is a learned scalar (per head, per layer) initialized ~0.5–0.8. The negative map `A₂` is trained (via the LM loss) to absorb the noise present in `A₁`; subtracting it leaves the sharpened signal.

Halving Q and K keeps the parameter budget equal to standard MHA — differential attention is a *rearrangement*, not an expansion.

## Why it matters

- **Sharper attention.** Empirically, attention entropy drops substantially — more mass concentrates on the tokens that actually matter for the output.
- **Better long-context recall.** Needle-in-a-haystack tasks show consistent gains vs standard MHA at matched compute.
- **Reduced hallucination.** Because attention no longer leaks mass to irrelevant tokens, the model conditions more cleanly on the actually-relevant context.
- **Composable with cache-compression techniques.** The subtraction can operate on MLA-compressed KV representations — this is exactly what Motif 3's GDLA does.

## Gotchas & tricks

- **λ scheduling matters.** Free-parameter λ can grow unbounded and dominate the first term; init near 0.5 and clip during training if instability appears.
- **Requires more attention FLOPs than vanilla MHA.** Two attention maps means two softmax + matmul operations; per-head compute roughly doubles even though parameter count doesn't.
- **The "noise" isn't purely bad.** A₂ absorbs whatever the training signal marks as noise for the current task; on some tasks (e.g. broad-context aggregation), the noise map may erroneously absorb useful mass.
- **Interacts with QK-norm.** Both techniques sharpen attention; combining them can over-sharpen. Tune QK-norm scale downward when stacking with differential attention.
- **Cache compression compatibility.** Naïvely, having two K matrices means twice the KV cache. Fixing this requires either sharing K₁ and K₂ (loses expressiveness) or compressing both into a shared latent — the GDLA path.

## Sources

- Paper: *Differential Transformer* — Microsoft Research, 2024 — the original method and empirical results.
- Related: [gdla.md](gdla.md) — Motif 3's fusion of differential attention with MLA's compressed KV.
- Related: [qk-norm.md](qk-norm.md) — an alternative attention-sharpening lever.
