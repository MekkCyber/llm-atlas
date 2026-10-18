# Virtual Head Attention (VHA)
*Depth — attention variant with linear training-time head expansion that folds into GQA at inference.*

**TL;DR:** VHA halves the number of physical query heads and uses only two KV heads (GQA-2 shape), but adds two small *linear* layers during training: Q Premix expands query heads into "virtual" heads, and Linear Postmix mixes features across heads. Because both are linear, matrix absorption collapses them into the standard Q and output projections at inference — the deployed model runs as vanilla GQA-2 with no extra parameters or compute. Reported wins at every scale from 0.6B to 8B-A1B and a 30B-A3B MoE, with smaller KV cache and fewer training FLOPs.

**Prereqs:** [multi-head-attention.md](multi-head-attention.md), [attention.md](../fundamentals/attention.md)
**Related:** [mla.md](mla.md), [transformer-block.md](transformer-block.md), [qk-norm.md](qk-norm.md)

---

## What it is

A training-side modification that gives GQA-2 inference approximately MHA-quality representation. The trick is that virtual heads only matter during the forward/backward pass; at inference they're folded away.

## How it works

- Physical layout: `H/2` query heads, 2 KV heads (GQA-2), training.
- **Q Premix.** A linear map `W_pre: ℝ^{d_q·(H/2)} → ℝ^{d_q·H}` expands the `H/2` physical query heads into `H` virtual heads before attention.
- **Linear Postmix.** A linear map `W_post: ℝ^{d_v·H} → ℝ^{d_v·(H/2)}` mixes virtual-head outputs back down.
- At inference, absorb `W_pre` into the Q projection and `W_post` into the output projection — one matmul each. The compute graph becomes identical to standard GQA-2.
- Loss sees `H` virtual heads; serving sees `H/2` physical heads.

## Why it matters

GQA-2 is the standard efficient-inference attention shape — half the KV cache of MHA, minimal quality loss. VHA says you can recover more of MHA's quality (via richer training-time computation) without paying for it at serving time. If it reproduces, it's a drop-in replacement for GQA in pretraining recipes.

## Gotchas & tricks

- The linearity constraint is strict: any non-linearity between query heads and the softmax breaks the fold. QK-norm with LayerNorm works (per-head stats absorbed); batched norms would not.
- Reported FLOP savings in training come from the smaller physical KV heads; the premix/postmix compute is small but non-zero.
- Reported comparisons are vs. GQA baselines, not MHA — VHA's claim is "better than GQA-2 at same serving cost," not "equal to MHA."

## Sources

- Paper: *Virtual Head Attention* — Ding, Wang et al., Baidu Inc. / Peking University, 2026 — [arXiv:2610.17959](https://arxiv.org/abs/2610.17959)
