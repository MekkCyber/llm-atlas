# Hybrid Linear–Softmax Attention

*Depth — mix gated linear attention layers with periodic softmax "anchor" layers to keep O(N) cost and full-rank expressiveness.*

**TL;DR:** Pure linear-attention transformers scale linearly with sequence length but lose full-rank token interaction — long-range dependencies degrade. Pure softmax attention has full-rank interactions but O(N²) cost. A hybrid stack alternates *many gated linear attention blocks* with *a few softmax anchor blocks* (e.g. 3:1 ratio) so most compute stays linear while the anchors periodically refresh full-rank structure. **Block Attention Residuals (AttnRes)** further route anchor outputs into downstream linear layers so their representations don't drift out of full rank between anchors. Introduced at scale in **SANA-Video 2.0** (NVIDIA, 2026).

**Prereqs:** [multi-head-attention.md](./multi-head-attention.md), [transformer-block.md](./transformer-block.md)
**Related:** [mla.md](./mla.md) · [_moe.md](./_moe.md)

---

## What it is

The attention-variants space has three broad answers to the O(N²) cost of softmax:

| Family | Cost | Full-rank interactions | Example |
| --- | --- | --- | --- |
| Softmax attention | O(N²) | yes | vanilla Transformer, MHA |
| Linear attention (kernelised, gated) | O(N) | no — rank capped by feature dim | RWKV, gated linear attn, Mamba-style |
| **Hybrid linear–softmax** | ~O(N), softmax fraction $s$ | yes, refreshed every $1/s$ layers | SANA-Video 2.0, Jamba-style |

The hybrid is *not* a compression of a pretrained softmax model. It is trained from scratch so the model learns the coordination between the two block types end-to-end.

## How it works

Two mechanisms, stacked:

1. **Alternating stack.** Each transformer block is one of two types — a gated linear attention block or a softmax "anchor" block — arranged in a fixed pattern (SANA-Video 2.0: 3 linear : 1 softmax). Anchors are the O(N²) sites; everything else is O(N).

2. **Block Attention Residuals (AttnRes).** After an anchor block produces its output, a summary of that output is *routed forward* into the linear-attention blocks that follow, on top of the usual residual stream. This lets downstream linear layers reuse the anchor's full-rank context, lifting deep-layer effective rank (SANA-Video 2.0 measures ~12% gain).

The linear-attention blocks retain a gating mechanism so they can amplify or suppress the anchor-injected feature depending on token content, avoiding forcing every downstream computation through the same anchor.

## Why it matters

- **Softmax-quality at ~linear cost.** VBench 84.30 for SANA-Video 2.0 at 5B / 13.2s per 480p clip on a single H100, competitive with far larger softmax video DiTs, with a compiled forward pass 3.2× faster than a matched full-softmax baseline at 720p/60s.
- **Long-sequence scaling comes back into reach.** Video generation, long-document reasoning, and other long-context settings have been quadratic-attention bound; the hybrid unbounds them without swapping to a lower-quality regime.
- **From-scratch, not linearised.** Linearising a pretrained softmax model has been the field's default cheap route to linear cost, but it caps quality below the pretrained ceiling. Training the hybrid natively means the two block types learn to specialise (anchors for structure, linear for local mixing) rather than one being forced to imitate the other.

## Gotchas & tricks

- **The mix ratio is a first-order hyperparameter.** SANA-Video 2.0 fixes 25% softmax as the quality/efficiency sweet spot via reduced-resolution proxy studies; changing this changes both quality and compiled-kernel throughput non-trivially.
- **AttnRes routing must respect the residual stream.** Naively summing an anchor summary into every downstream layer will double-count. The paper injects at specific layer indices — the exact schedule is part of the recipe.
- **Kernel fusion is where the speedup lives.** Standalone linear-attention layers are 2–3× the theoretical floor without fused kernels; SANA's Sol-Engine reports another 3.58× from kernel fusion + caching + sparse-attention tricks.
- **Not obviously portable to language pretraining.** Video DiTs run at a very different regime (long spatial-temporal sequences, few autoregressive steps). Whether the same 3:1 ratio and AttnRes routing transfers to autoregressive LLMs is untested.
- **Anchors are the calibration surface for evaluation.** If the anchors are too infrequent, deep layers under-mix; if too frequent, you lose the linear-cost advantage. The 3:1 ratio was tuned on reduced-resolution proxies — a domain-specific search is likely needed for other modalities.

## Sources

- Paper: *SANA-Video 2.0: Hybrid Linear Attention with Attention Residuals for Efficient Video Generation* — Chen, Yu, Li, Xue, Liu, Xin, Zhao, Ye, Wu, Wang, Zhou, Luo, Han, Xie — NVIDIA, 2026 — introduces the 3:1 hybrid stack and Block Attention Residuals.
- Related lineage: Jamba (attention/SSM hybrid), MEGA, and Griffin — earlier hybrids in the LLM setting with different mixing patterns.
