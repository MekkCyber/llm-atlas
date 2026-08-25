# Training-Free Sparse Attention
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Replace dense attention at inference time with a sparse operator that only computes a subset of query-key pairs, without retraining. The two hard sub-problems are (i) **which pairs to keep** (partition/routing) and (ii) **how to reconstruct the missing residual** so post-softmax outputs match dense. **SparsePR** (Taghavi et al., 2026) combines Response-Coupled Partitioning with Probe-Fitted Residual Reconstruction, delivering 1.48–2.61× speedups on video generators and world models at ~25% executed-pair density.

**Prereqs:** [../fundamentals/attention.md](../fundamentals/attention.md)
**Related:** [../fundamentals/dca.md](../fundamentals/dca.md)

---

## What it is

Attention cost is `O(L · L)` in sequence length. Video diffusion transformers push `L` into the hundreds of thousands (frames × spatial tokens), making attention the dominant cost. **Training-free** sparse attention swaps in a sparse operator *at inference time only* — the base model weights are unchanged. Contrast with training-time sparse attention (Longformer, BigBird) which requires retraining.

## How it works

A training-free sparse attention operator decomposes into three moves:

1. **Partition/route.** Assign queries to key groups. Bad choice → co-routed queries have disjoint supports and everyone's outputs are wrong.
2. **Compute sparse attention.** Standard softmax attention restricted to the routed (Q_group, K_group, V_group) tiles.
3. **Reconstruct residual.** Estimate what got dropped and correct for it.

SparsePR's two-part contribution:

**Response-Coupled Partitioning.** Sample a small set of queries; for each sampled query, compute its full key-response vector. Cluster these vectors — queries with similar key-response profiles share routes. Centroids become the routing coordinates. This ensures co-routed queries actually have overlapping supports (unlike index-adjacency routing, which is the naive default).

**Probe-Fitted Residual Reconstruction.** Sample a small "probe" set of queries and compute their *exact* attention outputs. Fit a **call-specific affine correction** from the sparse-output → dense-output residuals in the probe set. Apply the same correction to the rest of the sparse outputs. The correction is affine in the output subspace observed on probes.

Both parts are done at each attention call — no offline calibration, no per-model tuning.

## Why it matters

- **Deployable acceleration.** No retraining, no weight changes, so it drops into any inference stack (vLLM, SGLang, Diffusers).
- **Video/world-model regime.** At contexts of 100K+ tokens, dense attention dominates end-to-end latency; a 2× speedup at preserved quality is the difference between playable and unplayable.
- **Isolates the two subproblems.** The paper's ablations show probe fitting accounts for most of the quality preservation; response-coupled partitioning provides additional headroom under a finite probe budget. Useful decomposition for future work.

## Gotchas & tricks

- **Probe budget is a knob.** More probes = better affine correction, worse speedup. Sweet spot is model-dependent; SparsePR reports 22–26% executed-pair density as the operational point.
- **Not for training.** These operators break gradients (the affine correction is derived from the same forward pass it's used in). Use training-time sparse attention (Longformer, sliding window, BigBird) if you need gradients.
- **Cluster count matters.** Too few clusters → coarse routing, high dropped-pair error. Too many → probe budget wasted on cluster-size overhead.
- **Interacts with KV cache format.** Paged / grouped KV cache layouts must be respected by the sparse partition; naively clustering across pages defeats cache locality.
- **Diffusion vs autoregressive.** Video diffusion uses full bidirectional attention (no causal mask) — SparsePR's setting. For causal LLM decoding, sparsification patterns look different (see e.g. sparse-window + retrieval hybrids).

## Sources

- Paper: *Partition the Support, Reconstruct the Residual: Training-Free Sparse Attention for Video Generation and World Models* — Taghavi, Langari, Pandey, 2026 — introduces SparsePR.
- Paper: *Longformer* — Beltagy et al., 2020 — training-time sparse attention baseline.
- Paper: *Efficient Streaming Language Models with Attention Sinks* — Xiao et al., 2023 — training-free attention modification for LLM decoding.
