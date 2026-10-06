# Triadic Linear Attention
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Linear attention compresses history into a *matrix* hidden state via an outer product of key and value. Triadic linear attention generalizes to a *third-order tensor* state built from the outer product of two keys and a value, read by contracting both key axes with two queries. An $E$-dimensional second key multiplies state size by $E$ while adding only two projections. Applied to Gated DeltaNet and scalar-gated linear attention, it substantially improves long-context modeling and recall. Introduced in *Triadic Linear Attention* (2026, arXiv:2609.36529).

**Prereqs:** [../fundamentals/attention.md](../fundamentals/attention.md), [transformer-block.md](transformer-block.md)
**Related:** [README.md](README.md), [../fundamentals/dca.md](../fundamentals/dca.md)

---

## What it is

Linear attention's state is $S \in \mathbb{R}^{d_k \times d_v}$ — a matrix. The write is $S \mathrel{+}= k \otimes v$ (outer product); the read is $o = q^\top S$. The state's capacity is $d_k \cdot d_v$, so growing capacity means growing one of the per-token dimensions.

Triadic linear attention replaces the matrix with a *3-tensor* $T \in \mathbb{R}^{d_{k_1} \times d_{k_2} \times d_v}$. The write is $T \mathrel{+}= k_1 \otimes k_2 \otimes v$; the read is a double contraction $o = q_1^\top \, T \, q_2$ (schematically). An $E$-dim second key axis multiplies state capacity by $E$ while adding only two extra projections ($k_2$ and $q_2$) — far cheaper than enlarging either $d_k$ or $d_v$.

## How it works

Write operation per token:
$$
T_{t} = T_{t-1} + k_1^{(t)} \otimes k_2^{(t)} \otimes v^{(t)}
$$

Read operation:
$$
o^{(t)} = \mathrm{einsum}(\text{"ijl,i,j->l"},\; T_t,\; q_1^{(t)},\; q_2^{(t)})
$$

The construction is compatible with the standard linear-attention toolbox:

- **Data-dependent forgetting.** Multiplicative gates on $T$ work unchanged.
- **Delta rule.** The DeltaNet update replaces the plain write with a correction; this transfers to the 3-tensor by applying the correction over one key axis.
- **Chunkwise parallel training.** The chunk formulation generalizes — chunks compute partial 3-tensors and merge via the same associative-scan trick.

Applied to Gated DeltaNet and scalar-gated linear attention as drop-in replacements for the matrix state.

## Why it matters

Linear attention has been catching softmax attention in long-context modeling, bottlenecked by state capacity. Prior work spent parameters to enlarge the state (wider $d_k$, wider $d_v$, more heads). Triadic reframes capacity growth as a *structural* change: add a key axis, multiply capacity by its dimension, pay only two projections.

Reported: substantial improvements over matrix-state linear attention on MQAR, PG19 chunked loss, and bench recall — at the same parameter budget — and outperforms alternatives that enlarge the state through other mechanisms.

## Gotchas & tricks

- **Memory cost scales with $d_{k_2}$.** The state is now $d_{k_1} \cdot d_{k_2} \cdot d_v$. Picking a modest $d_{k_2}$ (e.g. 16-32) gives most of the recall gain at manageable memory.
- **Associativity requires care in chunkwise training.** The 3-way outer product is still associative under +, but efficient chunk kernels need custom einsum paths; naive implementations are slow.
- **Transfers across linear-attention variants but not softmax.** The construction depends on the linear write-as-outer-product shape. Softmax attention has no corresponding state to enlarge.
- **Not a replacement for data-dependent gating.** The paper shows triadic + gating together are stronger than either alone; the state-size and forgetting axes are complementary.

## Sources

- Paper: *Triadic Linear Attention: Three-Dimensional Recurrent States for Long-Context Sequence Modeling* — Sieberling, Runwal, Jin et al., 2026 — [arXiv:2609.36529](https://arxiv.org/abs/2609.36529).
