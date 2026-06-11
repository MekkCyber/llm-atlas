# DSA for multimodal (Lightning Indexer + GQA Sparse Aggregation)

*Depth — DeepSeek Sparse Attention adapted to GQA-based multimodal stacks for lossless 256K context.*

**TL;DR:** DeepSeek Sparse Attention (DSA) in text models picks a per-token Top-k sparse index set with an MQA-style "Lightning Indexer" and runs sparse attention over only those k positions. Keye-VL-2.0 (2026) adapts this to a GQA backbone for *multimodal* video+image+text input: the indexer remains MQA-shaped (one set of index scores per token) and the sparse aggregation step reuses the same index set across all GQA groups. Brings attention complexity from O(L²) to O(L·k) with k≈2048, enabling lossless 256K video context without dropping frames.

**Prereqs:** [../architectures/multi-head-attention](../architectures/multi-head-attention.md), [../fundamentals/attention](../fundamentals/attention.md)
**Related:** [../architectures/mla](../architectures/mla.md) · [README](README.md)

---

## What it is

Standard attention scales as O(L²) — fatal for hour-long video where L can easily reach 200K+ tokens. Prior fixes (sliding windows, sparse patterns, linear attention, MoBA) all degrade quality somewhere. DeepSeek Sparse Attention is the 2025 design that learns *which* k positions matter per query and attends only there.

Multimodal adoption is non-trivial: most production multimodal stacks use GQA (groups of query heads sharing K/V) rather than the MHA pattern DSA was designed against. Keye-VL-2.0's contribution is the MQA-indexer + GQA-sparse-aggregation pattern that respects GQA's KV sharing while keeping DSA's per-token Top-k sparsity.

## How it works

### Lightning Indexer (MQA-style)

For each query token, compute a *global index score* against every key position using a lightweight MQA pattern (one score per (query, key-position) pair, shared across heads):

$$
\text{score}_{q,p} = \mathrm{indexer}(Q_q, K_p) \quad \in \mathbb{R}
$$

The indexer is a small parametric function (e.g. a tiny attention head) that runs in O(L) per query rather than O(L·d) of a full key match. Take the top-k highest-scoring positions per query.

### GQA Sparse Aggregation

Standard GQA partitions $H$ query heads into $G$ groups, with each group sharing one K and one V. DSA-for-GQA does sparse attention as follows:

1. The MQA-style indexer produces *one* sparse index set per query token (independent of which GQA group the head is in).
2. All $H$ query heads (across all $G$ groups) aggregate from the *same* sparse subset of positions, using their group's K/V.
3. Heads in the same group share their group's K/V at the chosen sparse positions; heads across groups use different K/V at those positions.

Reusing the index set across groups is what makes the kernel implementation efficient — you only fetch the k positions once per token, then run all H heads on that buffer.

### Asymptotic cost

For sequence length L, head dim $d_h$, $H$ heads, top-k = k:

- **Dense:** $O(L^2 \cdot H \cdot d_h)$.
- **DSA-GQA:** $O(L \cdot k \cdot H \cdot d_h)$ for the sparse attention + $O(L^2)$ for the indexer (small constant).

With k = 2048 and L = 256K, this is roughly 128× less work for the dominant attention term.

### Two-stage training

The paper trains DSA-multimodal in two stages:
1. **Dense warm-up.** Train the indexer to predict the dense-attention pattern — supervised by the original full-attention scores. Aligns the indexer with the model's "natural" attention.
2. **Sparse adaptation.** Switch to Top-k sparse attention and continue training the whole model end-to-end. The indexer now drives real routing decisions; the model adapts to operate on the sparse set.

## Why it matters

- **Hour-scale video becomes loss-less.** Earlier long-video stacks dropped frames or pooled aggressively; DSA-multimodal keeps the full 256K context active.
- **GQA stays.** Production multimodal backbones use GQA for KV-cache reasons; DSA-multimodal preserves that without reverting to MHA.
- **Generalizes beyond video.** The same pattern works on long-document multimodal (interleaved image + text), long-context agentic transcripts, and any multimodal input where O(L²) is the bottleneck.

## Gotchas & tricks

- **k matters more than L.** Smaller k gives more speedup but reduces effective attention bandwidth. k = 2048 is the paper's choice; degrades quality below ~1K, wastes compute above ~4K.
- **Indexer quality is the failure mode.** A bad indexer routes attention to the wrong positions and the sparse approximation collapses. The dense warm-up stage exists to prevent this.
- **Custom kernels are mandatory.** A naive implementation that materializes the Top-k mask burns the speedup. The paper notes custom kernels are part of the contribution.
- **GQA group count constrains the pattern.** Heavily-grouped GQA (e.g. 8 heads per group) means many heads share the same K/V at the chosen sparse positions, reducing the model's ability to diversify across heads. Light grouping (2–4 per group) is closer to MHA semantics.
- **Indexer + sparse aggregation must agree on positions.** Implementations that compute the indexer asynchronously and then aggregate can mis-align — design-time choice.

## Sources

- Paper: *Kwai Keye-VL-2.0 Technical Report* — Wen et al., Kwai/Kuaishou, 2026 — [arXiv 2606.10651](https://arxiv.org/abs/2606.10651).
- Background: *DeepSeek-V3.2: Sparse Attention* — DeepSeek, 2025 — DSA in text-only LLMs.
- Background: *GQA: Training Generalized Multi-Query Transformer Models* — Ainslie et al., 2023.
