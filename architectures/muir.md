# Muir: Sensitivity-Allocated Gating + Matched-Filter Readout Linear Attention
*Depth — a linear-attention block that closes the gap to Transformer by redesigning the gate and the readout.*

**TL;DR:** Muir is a Gated-DeltaNet-2 variant that (1) replaces the full-rank input gate with a **sensitivity-allocated low-rank projection**, (2) adds a **matched-filter readout gate** that re-weights the recurrent state's rows by their correlation with the query, and (3) recycles the parameter savings into a wider FFN. A boundary-pivoted chunkwise training kernel keeps the pass efficient on wide-matrix accelerators. The result is strictly better than GDN-2 and Transformer at matched compute from 166M to 1.34B parameters.

**Prereqs:** [attention](../fundamentals/attention.md), [transformer-block](transformer-block.md), [multi-head-attention](multi-head-attention.md)
**Related:** [mla](mla.md), [_moe](_moe.md)

---

## What it is

Gated-DeltaNet (GDN) is a modern linear-attention recurrence: each token updates a state matrix $S_t$ by a rank-1 outer-product plus a gated carry. Muir sits in the same family but changes two components whose cost dominates quality at matched params:

- the **input gate** that controls how much of the new (key, value) pair enters the state,
- the **readout** that produces the output given a query.

Saving parameters on the gate buys headroom for the FFN, where modern scaling is more parameter-hungry than attention.

## How it works

### Sensitivity-allocated gating

Standard GDN-2 uses a full gate matrix $G \in \mathbb{R}^{d \times d}$. Muir factors it as $G = U \mathrm{diag}(s) V^\top$ with $s \in \mathbb{R}^r$, $r \ll d$. The twist is that $s$ is **allocated by estimated sensitivity**: rank-1 components whose directions the forward pass is most sensitive to get larger $s_i$, learned via a sensitivity-weighted penalty at init. This keeps only the ranks that move the loss and leaves the rest near-zero without a hard rank-r cutoff.

### Matched-filter readout gate

At query time, the output is

$$
y_t = \text{diag}\!\left( \sigma(q_t^\top S_t) \right) \cdot S_t \cdot q_t
$$

— each row of $S_t$ is scaled by how much its *key component* correlates with $q_t$ before the final matmul. This is a cheap per-row attention scoring on top of the recurrent state — it recovers some of the content-addressable retrieval that pure recurrences lack.

### Boundary-pivoted chunkwise training

GDN-style recurrences are trained with chunkwise parallel passes: split the sequence into chunks, do intra-chunk attention-style passes, then cross-chunk recurrence. Muir pivots the chunk boundary into the parallel pass so most chunks fit a single wide matrix unit (TPU MXU, H100 SXM tile), reducing kernel-launch overhead.

### FFN budget

Parameters saved on the gate are moved to the FFN: GDN-2 and Muir are matched on total params by widening the Muir FFN.

## Why it matters

- **Narrows the linear-attention gap.** At 166M → 1.34B params, Muir reports lower perplexity than GDN-2 and than Transformer at matched compute, plus gains on commonsense, recall and long-context retrieval (32K).
- **Long-context training is faster than Transformer.** Chunkwise kernels + linear-time attention means 32K training is cheaper than quadratic-attention Transformers at the same width.
- **Architectural lesson:** when a linear-attention block underperforms, the fix is as likely to be in the gate / readout as in the recurrence itself.

## Gotchas & tricks

- **Sensitivity estimation is at init.** The allocation is not updated during training; if your distribution shifts significantly post-mid-training the rank allocation can become stale.
- **Readout cost is non-negligible.** The matched-filter gate adds an $O(d \cdot \text{state rows})$ scoring per token. Cheap vs quadratic attention, not free vs a bare recurrence.
- **Boundary-pivoted kernel is hardware-sensitive.** Tuned for TPU/H100 wide matrix units; on smaller accelerators (A100, consumer GPUs) the win can shrink.
- **Linear-attention retrieval is not free.** The matched-filter readout helps at long context but is not a substitute for full attention on tasks that need exact token-level copying from arbitrary positions.

## Sources

- Paper: *Muir: Linear Attention with Sensitivity-Allocated Gating and Matched-Filter Readout* — Tabak, Schaefer, et al., 2026 — [arXiv:2611.08376](https://arxiv.org/abs/2611.08376).
- Background: *Gated DeltaNet* — Yang et al., 2024 — the GDN / GDN-2 baseline.
- Background: *Mamba* — Gu & Dao, 2023 — the broader state-space family Muir sits in.
