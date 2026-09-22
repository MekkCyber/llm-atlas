# Mixture-of-Memory embeddings
*Depth — context-aware sparse parametric memory as a companion class to MoE.*

**TL;DR:** *Value Embedding* and related methods add a cheap parametric memory to a transformer as an extra token-indexed lookup table — each token id maps to one memory row that's added to the hidden state. This is fast (one lookup per token) but *context-agnostic*: polysemous tokens ("python" the language vs. the animal) collide on the same memory row. Mixture-of-Memory (MoME, Li et al., 2026) replaces each row with a mixture of M slots and uses a learned gate over the hidden state to pick which slots to read. Same look-up cost per slot, contextual disambiguation for free, and better iso-parameter / iso-FLOP scaling than baseline value-embedding methods at sub-billion scale.

**Prereqs:** [attention.md](../fundamentals/attention.md), [_moe.md](./_moe.md)
**Related:** [aux-loss-free-balancing.md](./aux-loss-free-balancing.md) · [deepseek-moe.md](./deepseek-moe.md)

---

## What it is

A sparse-capacity mechanism *distinct from MoE*: it augments the backbone with a cheap parametric memory that is queried *per token* rather than routed *per expert FFN*. The "memory" is a large embedding table whose entries are added into (or otherwise mixed with) the hidden state at each position.

MoME's contribution is to make this memory **contextual** — same surface token, different memory row, depending on the local hidden state.

## How it works

For a token at position $t$ with surface id $x_t$ and hidden state $h_t$:

1. **Slot bank.** Instead of a single memory row $M[x_t]$, the table stores M slots: $\{M_1[x_t], M_2[x_t], \ldots, M_M[x_t]\}$.
2. **Gate.** A small linear map (or MLP) on $h_t$ produces gating scores over the M slots: $g_t = \mathrm{softmax}(W_g h_t)$.
3. **Mixture read.** The memory contribution is $\sum_{i=1}^M g_{t,i} \cdot M_i[x_t]$ — or, more commonly, a top-$K$ sparse variant that reads only the highest-scoring slots.
4. **Injection.** The read value is added to $h_t$ (or into an FFN input; both variants are studied).

The gate is *content-conditioned*: two occurrences of "python" in different contexts route to different slots, giving each sense its own row.

## Why it matters

- **Sparse capacity, distinct pattern from MoE.** MoE routes to expert FFNs (whole subnetworks); MoME routes to memory rows (single vectors). MoME reads scale as $O(M)$ additions per token, far cheaper than firing M FFNs.
- **Fixes a real defect in Value Embedding.** Collision on polysemous tokens is a known source of representation interference. Qualitative routing analyses in the paper show clean sense-disambiguation on words like "python" — the gate isn't just noise.
- **Iso-parameter and iso-FLOP wins.** On nanochat, Llama-3 / MobileLLM, and Qwen3 backbones (sub-billion scale), MoME beats Value Embedding, Bigram, and STEM baselines at matched parameters and matched training FLOPs, with a favorable memory-size scaling trend.
- **CPU / edge friendly.** Because reads are lookups + adds (not GEMMs), memory-augmented models are attractive when the compute budget is small but memory bandwidth is available — a very different deployment regime from MoE.

## Gotchas & tricks

- **Slot balance is a live issue.** Nothing in the recipe forces slots to be used uniformly. A collapsed gate that always picks slot 1 degenerates back to Value Embedding. Load-balancing hooks similar to MoE ([aux-loss-free-balancing.md](./aux-loss-free-balancing.md)) may be needed at larger scale.
- **Sparse-read on the memory side, not the gate side.** The gate is small; the memory table is huge. Read patterns must exploit slot sparsity or throughput collapses.
- **Only sub-billion evaluated.** The paper's evidence is at sub-billion params. Scaling behavior at LLM scale is speculative; the "memory-size scaling trend" is favorable but extrapolated.
- **Not a drop-in replacement for MoE.** MoE and MoME solve different bottlenecks: MoE grows FFN capacity, MoME grows *lookup* capacity. They can stack (MoE FFNs + MoME memory) but haven't been jointly studied.

## Sources

- Paper: *MoME: Mixture-of-Memory Embeddings for Context-Aware Sparse Lookup* — Li, Sigal, Liao, University of British Columbia, 2026 — [arXiv:2609.15126](https://arxiv.org/abs/2609.15126).
- Related: Value Embedding, Bigram, STEM (baselines cited in the paper).
