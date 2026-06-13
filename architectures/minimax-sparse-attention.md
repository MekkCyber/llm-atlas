# MiniMax Sparse Attention (MSA)

*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A blockwise sparse attention layer built on top of Grouped Query Attention. Each query token attends to a learned, content-dependent subset of key blocks rather than the full sequence. Paired with a custom GPU kernel that iterates only over selected blocks, MSA matches GQA quality while reducing per-token attention compute by **28.4×** at 1M-token context, delivering **14.2× prefill** and **7.6× decoding** wall-clock speedups on H800.

**Prereqs:** [multi-head-attention](multi-head-attention.md), [mla](mla.md), [attention](../fundamentals/attention.md)
**Related:** [_sparse-attention](_sparse-attention.md), [transformer-block](transformer-block.md)

---

## What it is

MSA replaces the dense softmax-attention step in a GQA-based LLM with a block-sparse variant. Tokens are grouped into fixed-size blocks. For each query token, a learned selector picks a small set of relevant key blocks, and softmax is computed only over those blocks. The remaining QK pairs are skipped entirely — not just masked after the fact, which is the trap that kills naive "sparse attention" implementations on GPU.

The point of building on GQA (rather than vanilla MHA) is that the KV-cache savings of GQA stack with the FLOP savings of sparsity. The point of building a custom kernel is that GPU softmax already costs $O(N^2)$ if the mask isn't realized at the iteration level.

## How it works

For a query token $q_t$ at layer $\ell$:

1. **Block partition** the key/value sequence into blocks of fixed size $B$.
2. **Selector**: a lightweight scoring function (the paper uses a low-rank query-to-block compatibility score) emits, per query head, the top-$k$ block indices. Selection is learned end-to-end and trained alongside the rest of the model.
3. **Block-sparse softmax**: the custom kernel iterates over only the selected blocks for each query, computes the dot product and softmax over that reduced set, and accumulates the value vectors. No full $N \times N$ score matrix is ever materialized.
4. **GQA grouping**: as in GQA, $H$ query heads share a smaller set of KV heads, so the KV cache per layer is the GQA-sized one, not the MHA-sized one.

The result: per-token attention FLOPs scale with $k \cdot B$ (the selected key budget), not $N$.

## Why it matters

Long-context inference is the dominant workload trend for agentic systems and persistent-memory LLMs. KV-cache compression alone (MLA / GQA) handles the memory side, but per-token attention *compute* keeps growing with context length under any softmax-style baseline. MSA targets the compute side directly, with a co-designed kernel so the savings show up in wall-clock terms. A 14.2× prefill speedup at 1M context fundamentally changes the deployment economics for repo-scale code reasoning and long-running agent loops.

## Gotchas & tricks

- **Selector quality is the failure mode.** A poorly trained selector either misses the right blocks (quality drop) or selects too uniformly (no FLOP win). Reported parity with GQA quality is the headline; expect to fine-tune the selector loss in practice.
- **The kernel matters as much as the algorithm.** "Sparse attention" implementations that mask after the dense softmax pay full compute. The block structure is chosen partly so that GPU tiles are kernel-friendly.
- **Stacks with MLA / GQA**, not a replacement for either. MSA reduces compute; MLA reduces KV memory; GQA reduces KV memory per head. Frontier long-context stacks combine all three.

## Sources

- Paper: MiniMax Sparse Attention — Xu et al. (2026) — [arXiv:2606.13392](https://arxiv.org/abs/2606.13392)
