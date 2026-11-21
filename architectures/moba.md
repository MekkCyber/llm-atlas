# Mixture of Block Attention (MoBA)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A trained sparse-attention mechanism that **routes each query to its top-$k$ KV blocks**, a transformer-native analogue of mixture-of-experts applied to attention instead of FFNs. Shipped in Kimi K1.5 and derivatives, it holds up to 128k+ context at dense-model quality while cutting attention FLOPs roughly linearly with $k/\text{N-blocks}$. The decode kernel's current acceleration frontier is **LightningMoBA** (dynamic threshold routing + KV-centric scheduling).

**Prereqs:** [multi-head-attention.md](multi-head-attention.md), [_sparse-attention.md](_sparse-attention.md)
**Related:** [../inference/lightning-moba.md](../inference/lightning-moba.md), [mla.md](mla.md), [../case-studies/kimi-k1-5.md](../case-studies/kimi-k1-5.md)

---

## What it is

Split the KV stream into contiguous blocks of fixed size (typically 64–128 tokens). For each query, a lightweight router scores every KV block; the query attends only to its top-$k$ blocks. Routing is end-to-end differentiable and trained jointly with the rest of the model — unlike post-hoc sparsity (MInference, Quest), MoBA's sparsity pattern is a trained property of the model.

## How it works

1. **Block the KV stream.** Partition $K, V \in \mathbb{R}^{T \times d}$ into $N = T/B$ blocks of size $B$.
2. **Block score.** Compute a per-block representative $\bar{K}_j \in \mathbb{R}^d$ (mean-pool of the block's keys, or a learned pooling). The router score for query $q_t$ on block $j$ is $s_{t,j} = q_t^\top \bar{K}_j$.
3. **Top-$k$ routing.** Attend to the $k$ highest-scoring blocks per query. Classical softmax attention runs on the selected blocks' full $K, V$. Blocks outside the top-$k$ contribute nothing to the output — the router's decision is hard, not a soft weighting.
4. **Causal mask.** Enforce causality by restricting the router's candidate set to past blocks.
5. **Training.** Gradient flows through the router via straight-through on the top-$k$ gate, plus an auxiliary load-balancing loss (analogous to MoE) to keep block utilisation even across heads.

### Shape at a glance

```
q_t ∈ ℝ^d
K  ∈ ℝ^{T×d}, V ∈ ℝ^{T×d}   split into N blocks of size B
block scores:  s_{t,j} = q_t · mean(K_j)       (one score per block)
selected:     top-k j indices
attention:    softmax over selected-block keys, output = weighted V
```

## Why it matters

- **Dense-quality long context at sub-quadratic FLOPs.** Kimi K1.5 trains at 128k with MoBA; attention cost scales with $k$, not $T$.
- **End-to-end trained sparsity.** The router learns which blocks matter; this consistently outperforms post-hoc sparsity schemes that keep the base model fixed.
- **Kernel-friendly.** Block granularity maps cleanly to GPU tiling, unlike token-level sparsity. [LightningMoBA](../inference/lightning-moba.md) further restructures the schedule to hit near-peak Hopper throughput.
- **Composes with hybrid designs.** Can live alongside full-attention layers in a model where recall-sensitive heads keep dense attention and the rest route — see [flashmorph2](flashmorph2.md).

## Gotchas & tricks

- **Block size vs $k$ tradeoff.** Small $B$ gives finer routing but a bigger router (more scores per query); large $B$ wastes compute inside the selected block. $B = 64$–$128$ is standard.
- **Router collapse.** Without the load-balance auxiliary loss, one head can collapse to attending to the most-recent block and freeze; aux loss analogous to MoE load-balancing is essential.
- **Needle-in-haystack queries.** A single-sentence needle can live entirely inside one block. If that block doesn't make the top-$k$, recall is 0. Mitigations: high-floor $k_{\min}$, per-head router diversity, landmark blocks.
- **Inference-time threshold routing.** [LightningMoBA](../inference/lightning-moba.md) shows that replacing fixed-$k$ routing with a calibrated threshold at inference improves both speed and quality — the trained top-$k$ was over-recruiting on easy queries.
- **Retrofitting dense models is non-trivial.** MoBA is trained in; grafting the router onto a dense pretrain usually needs long fine-tuning, and quality lags post-hoc sparsity methods designed for retrofits.

## Sources

- Paper: *MoBA: Mixture of Block Attention for Long-Context LLMs* — Kimi, 2025 — the original MoBA mechanism and training recipe.
- Paper: *Accelerating Mixture of Block Attention via Dynamic Routing, KV-Centric Execution and Runtime Pipelining* — Tencent, 2026 — see [lightning-moba](../inference/lightning-moba.md).
- Case study: [Kimi K1.5](../case-studies/kimi-k1-5.md) — the first frontier model shipping MoBA at long context.
