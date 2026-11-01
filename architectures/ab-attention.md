# AB-Attention: Accurate Block Attention
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A training-free sparse-attention method for video diffusion transformers. Three parts: a *selector* that scores small KV blocks cheaply via query clustering, a *sampler* that treats the chosen blocks as an importance-weighted estimator of dense attention rather than a truncation, and a tensor-core-aware *kernel* that packs small sampled blocks into full tiles. Skips 89–92% of attention ops at up to 2.23× end-to-end speedup, and *improves* PSNR by up to 4.4 dB over the strongest prior sparse-attention baseline on Wan2.1 and HunyuanVideo.

**Prereqs:** [multi-head-attention.md](multi-head-attention.md), [../fundamentals/attention.md](../fundamentals/attention.md)
**Related:** [../inference/README.md](../inference/README.md)

---

## What it is

Full 3D attention is the bottleneck in video diffusion transformers. Prior sparse-attention methods either (a) coarsen the block granularity and waste attention on near-zero tiles, or (b) use fine blocks that break tensor-core kernels. AB-Attention is a drop-in replacement that keeps fine selection *and* tensor-core-friendly compute by combining importance sampling with tile-packed kernels.

## How it works

**Step 1 — Selector.** For each query block, cluster queries into a small number of centroids. Score every candidate (small) KV block by its centroid-vs-key similarity; this makes scoring $O(\text{clusters} \times \text{small-blocks})$ instead of $O(\text{queries} \times \text{small-blocks})$.

**Step 2 — Sampler.** Instead of thresholding and keeping the top-$k$ blocks, sample blocks with probability proportional to score, then reweight the attention contribution by inverse selection probability:

$$
\mathrm{Attn} \approx \sum_{b \in S} \frac{1}{p_b}\,\mathrm{softmax}(Q K_b^\top) V_b
$$

where $S$ is the sampled subset and $p_b$ is block $b$'s selection probability. This yields an unbiased estimator of dense attention under mild assumptions — not a truncation.

**Step 3 — Kernel.** The sampled small blocks are packed into full tensor-core tiles at kernel launch time, so GPU-cycle cost scales with the *tile count* (coarse), not the *block count* (fine). Finer sparsity patterns become free in wall time.

Training-free: no weights change. Drop in at inference.

## Why it matters

The field had an implicit tradeoff between fine-grained sparsity (quality) and tensor-core friendliness (speed). AB-Attention breaks it by (a) reframing selection as estimation (unbiasedness) and (b) decoupling the sparsity pattern from the kernel's compute shape (packing). Expect ports to video, long-context text, and multimodal DiTs. The quality *improvement* (+4.4 dB PSNR vs. the strongest prior sparse baseline) is unusual — most sparse-attention papers lose quality to gain speed.

## Gotchas & tricks

- **Query-cluster count is the main knob.** Fewer clusters make selection cheaper but coarser; the paper tunes it per model.
- **Unbiasedness holds under softmax-attention assumptions.** The $1/p_b$ reweighting is correct for the attention sum, not for each row's softmax normalizer; the paper handles the normalizer separately.
- **Kernel packing requires at least one supported tile size.** Not all sparsity patterns pack cleanly into tensor-core tiles; the kernel enforces a minimum pattern granularity.
- **Benchmarks are on Wan2.1 and HunyuanVideo.** Transfer to text/code diffusion is plausible but not yet shown.
- **Not the same as flash-attention sparsity.** Flash variants reorder dense attention for memory efficiency; AB-Attention is a sparsified attention.

## Sources

- Paper: *AB-Attention: Accurate Block Attention for Accelerating Video Diffusion Transformers* — Xu, Zhang, Zhang, Yang, Ma, Wang, Wang, Zhou, Zhao — CUHK-Shenzhen et al., 2026 — [arXiv:2610.39663](https://arxiv.org/abs/2610.39663).
- Reference: *FlashAttention* — Dao et al., 2022 — dense tiled attention.
- Reference: *Wan2.1* and *HunyuanVideo* — the video-DiT baselines.
