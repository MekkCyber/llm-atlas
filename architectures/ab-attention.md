# AB-Attention

*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Accurate Block Attention. A training-free block-sparse attention for 3D video diffusion transformers that frames sparsity as *Monte Carlo estimation* of dense attention: score key/value blocks via intra-block query clustering, sample blocks proportionally to score, and reweight by inverse inclusion probability — one sparse pass gives an unbiased estimator of the dense output. On Wan2.1 and HunyuanVideo: 89–92% of attention skipped, +4.4 dB PSNR over strongest baseline, up to 2.23× end-to-end speedup.

**Prereqs:** [multi-head-attention](multi-head-attention.md), [../fundamentals/attention](../fundamentals/attention.md)
**Related:** [_moe](_moe.md), [capacity-factor](capacity-factor.md)

---

## What it is

Dense 3D attention is the single largest cost in modern video DiTs because the sequence length is `(frames × patches)` which blows up quadratically. Prior block-sparse schemes (e.g. selecting the top-$k$ blocks by a cheap score) are *biased* estimators of the dense output — at high sparsity they drop information unevenly, degrading fidelity. AB-Attention replaces the biased top-$k$ selection with an unbiased importance-sampling procedure that admits provably correct reweighting.

## How it works

For each query block $Q_i$:

1. **Score KV blocks.** Cluster the queries inside $Q_i$ (a few centroids per block). For each key/value block $K_j, V_j$, compute a score $s_{ij}$ from the centroid-level attention affinities — cheap because centroid count ≪ query count.
2. **Sample.** Draw a subset of blocks $S_i$ with inclusion probability $p_{ij} \propto s_{ij}$ (Horvitz–Thompson-style sampling).
3. **Weighted attention.** Compute attention only over the sampled blocks, dividing each attention contribution by $p_{ij}$:

$$
\hat A_i = \sum_{j \in S_i} \frac{1}{p_{ij}} \cdot \mathrm{softmax}(Q_i K_j^\top) V_j
$$

This is an **unbiased estimator** of the full dense attention output (standard IPW argument).

4. **Pack and kernel.** Sampled blocks per query block are rarely full-tile; a packed kernel groups them into contiguous tiles so the hardware actually sees dense matmuls on the sampled portion.

## Why it matters

- Video generation is bottlenecked on attention FLOPs; a training-free 2× end-to-end speedup is directly deployable on all existing video DiT checkpoints.
- The unbiasedness is the point: top-$k$ attention loses fidelity faster than its FLOP savings warrant, especially at >80% sparsity. IPW reweighting is a textbook fix that no prior work applied in this setting.
- Suggests a general recipe for inference-time attention sparsification across modalities: *importance-sample, don't threshold*.

## Gotchas & tricks

- The query-cluster count is the main hyperparameter. Too few clusters and scoring is coarse (blocks' importance underestimated); too many and the scoring itself becomes expensive.
- IPW variance can be high if $p_{ij}$ is near zero for important blocks; use a floor (`max(p_{ij}, ε)`) to clip variance.
- The packed kernel is important for actual speedup — a naive gather of sampled blocks gives most of the FLOP savings but little wall-clock win on GPU.
- Reported numbers are for video DiTs; text DiTs and autoregressive LMs may need different cluster-granularity defaults.
- Compatible with other inference optimizations (quantization, step distillation) but not with methods that already select a biased top-$k$ (the two estimators conflict).

## Sources

- Paper: *AB-Attention: Accurate Block Attention for Accelerating Video Diffusion Transformers* — Xu, Zhang, Zhang, Yang, Ma, Wang et al., 2026 — [arXiv:2610.39663](https://arxiv.org/abs/2610.39663).
- Prior: *Wan2.1* and *HunyuanVideo* — the video DiT baselines evaluated.
- Related: *Horvitz–Thompson (1952)* — the unbiased-sampling estimator this adapts.
