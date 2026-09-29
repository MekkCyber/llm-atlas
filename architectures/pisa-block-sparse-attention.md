# PISA: Pyramid Block-Sparse Attention
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Block-sparse attention lets a Transformer skip most $QK$ dot-products at long context, but *choosing* which blocks to keep is still $O(N^2)$ under naive top-K scoring. PISA replaces flat top-K with a hierarchical pyramid: pool keys into $O(\log N)$ coarser levels and progressively narrow the candidate set with LogSumExp scoring on bounded subsets, driving total selection cost to $O(N \log N)$.

**Prereqs:** [multi-head-attention](./multi-head-attention.md), [attention](../fundamentals/attention.md)
**Related:** [mla](./mla.md), [transformer-block](./transformer-block.md)

---

## What it is

Block-sparse attention partitions the key axis into blocks and, for each query, computes attention only against a small subset of "kept" blocks. This makes attention throughput sublinear in $N$ **if** you can pick the right blocks cheaply. Existing selectors (top-K over per-block max-inner-product, dynamic routing) still touch every block, so the *selection* step is $O(N^2)$ even when the *attention* step is not.

PISA reframes selection as a **coarse-to-fine pyramid search**:

- Build $L = O(\log N)$ levels of keys by successive pooling (mean- or max-pool along the sequence).
- At the coarsest level, score against a **bounded** candidate pool using LogSumExp — a soft-max-friendly proxy for "how much attention mass would this cell carry?".
- Keep the top few candidates; refine to the next finer level; score again against the small candidate set; repeat until you reach block granularity.

Because each level has bounded fan-out, the total number of scoring ops is $O(N \log N)$.

## How it works

For a query $q$:

1. **Coarsest level** ($L-1$). All pooled keys at this level fit in a small set $C_{L-1}$; compute $s = \mathrm{LSE}_i(q \cdot k_i^{(L-1)})$-style scores and take top-$K_{L-1}$.
2. **Refine.** For each surviving coarse candidate, expand into its children at level $L-2$; score again with LogSumExp on the (still bounded) candidate pool; keep top-$K_{L-2}$.
3. **Base level.** Continue to level 0 (individual key blocks). The final surviving set is what dense attention runs against.

Hardware-aware **Triton kernels** fuse routing and scoring so the full $QK$ score matrix is never materialized — that is what turns the $O(N \log N)$ arithmetic count into wall-clock speedup on modern GPUs.

The scheme is trained end-to-end: the router is fully differentiable (LogSumExp is smooth), so the model learns which coarse pools to trust.

## Why it matters

- **Removes the last quadratic term** in long-context inference. Sparse-attention selection was the remaining $N^2$ step in stacks that had already switched to sparse matmuls.
- **Drop-in for training and inference.** Kernels for both directions are published; integrates into a Transformer replacing dense self-attention.
- **Retrieval-heavy tasks improve.** PISA matches dense attention on commonsense reasoning benchmarks while beating flat top-K selectors on retrieval, where the pyramid's ability to zoom in on distant relevant blocks helps.

## Gotchas & tricks

- **Pooling choice matters.** Mean pool is smoother but blurs sharp positional signals; max pool preserves peaks but risks selecting on outliers. The paper favors LSE-style pooling that respects the attention statistic.
- **Bounded candidate pools require careful sizing.** Too small at coarse levels and you prune the right region early; too large and $O(N \log N)$ regresses toward quadratic. Tune per model.
- **Composes with MLA / GQA / KV compression.** PISA operates on the block-selection axis; it is orthogonal to per-head or per-layer memory tricks and can stack with them.
- **Training-time selection needs care.** During training, the sparse mask must not starve gradients from the true relevant blocks; the paper uses annealing (denser masks early, sparser later).
- **Long-context calibration.** The pyramid's benefit shows up at $N \gtrsim 16$K; at short context the extra scheduling is overhead.

## Sources

- Paper: *Block Sparse Attention with Log-Linear Complexity* — Bohao Tang, Zhen Qin, Yuqi Pan, Zheng Li, Pengfei Liu — ByteDance Seed, 2026 — https://arxiv.org/abs/2609.31093
- Related: *Native Sparse Attention* (DeepSeek), *MoBA* (Kimi), *Star Attention* — earlier sparse-selection schemes that PISA improves on.
