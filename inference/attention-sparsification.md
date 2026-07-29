# On-the-Fly Attention Sparsification
*Depth — runtime block-thresholded attention that trades accuracy-preserving speedups on long-sequence transformers with no training or calibration (Sol-Attn).*

**TL;DR:** At each attention step, dynamically threshold attention blocks and skip the sub-threshold ones — no offline calibration, no model retraining, no static mask. The pattern reflects the *current* query/key structure at that step and layer, so it adapts across prompts. On video diffusion transformers, Sol-Attn reports **2.1× / 2.3×** end-to-end speedups for generation / editing at preserved visual quality.

**Prereqs:** [../inference/README.md](../inference/README.md), [../fundamentals/attention.md](../fundamentals/attention.md)
**Related:** [../architectures/multi-head-attention.md](../architectures/multi-head-attention.md)

---

## What it is

For long-sequence transformers — high-resolution video-DiT, long-context LLMs, image editors — attention is the dominant inference cost because it scales quadratically in sequence length. Static sparse-attention patterns (windowed, block-sparse) cap the ceiling; learned sparsity needs retraining. On-the-fly sparsification computes a cheap block-level score at each step and drops blocks below a threshold, adapting to the sequence.

## How it works

Per layer, per step:

1. **Block score.** Group queries and keys into blocks. Compute a cheap approximation of `Q_block · K_block` — a max, a top-k row score, or a low-rank sketch — to estimate the block's contribution.
2. **Threshold.** Compare each block score against a threshold (fixed or fraction-of-max) and mark blocks below it as skip.
3. **Sparse compute.** Materialize the softmax and value read only for surviving blocks. The kept mask changes per step and per layer.

No calibration pass, no fine-tune, no mask stored on disk. The dynamic threshold means dense-attention behavior is recovered when the current QK actually needs it.

## Why it matters

Video-DiT and long-context transformer inference has been gated by attention cost. A **2×-plus** drop-in speedup at preserved quality brings video generation into the cost regime of image generation, and long-context serving down toward affordable per-token pricing. Complements paged attention, continuous batching, and speculative decoding — attacks a different piece of the wall clock (the attention kernel itself).

## Gotchas & tricks

- Block granularity trades headroom for approximation error. Smaller blocks are more selective but the score itself gets expensive.
- Threshold should be *relative* (fraction of the max block score) not absolute — absolute thresholds fail across prompts that produce different softmax scales.
- The pattern works because in practice most block scores are near zero — inspect the distribution before shipping to check the assumption holds for your model.
- Combines multiplicatively with FlashAttention-style kernels — the surviving blocks still benefit from fused attention.

## Sources

- Paper: *Sol-Attn: Accelerating Video Generation Inference via On-the-Fly Attention Sparsification* — Li et al., 2026 — [arXiv:2607.24027](https://arxiv.org/abs/2607.24027)
