# Grouped Query Experts (GQE)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A **mixture-of-experts layer on top of GQA self-attention**: within each GQA group, a router selects `k` of the query-head experts per token while all KV heads stay dense. The result is a sparse attention layer that preserves the KV-cache footprint of GQA but cuts query-side compute roughly in half. At 250M params / 30B tokens, GQE matches the all-active GQA baseline while activating only half the query-head experts per token.

**Prereqs:** [multi-head-attention.md](multi-head-attention.md), [_moe.md](_moe.md)
**Related:** [mla.md](mla.md), [_moe.md](_moe.md)

---

## What it is

GQA already shares KV heads across groups of query heads — a one-time reduction of KV-cache memory. GQE goes one step further on the *query* side: within a GQA group, the query heads become experts, and a router fires only a token-conditional subset of them.

The KV-cache benefit is untouched — the cache is keyed on KV heads, which remain dense and shared exactly as in GQA. The compute benefit is on query-head FLOPs and the per-head attention dot product.

## How it works

Per attention layer:

1. **Project hidden state** to `H_Q` query-head experts and `H_KV` shared KV heads (GQA layout).
2. **Routing.** For each token, a small router scores the `H_Q` query-head experts. Top-`k` are activated.
3. **Per-group activation.** Within each GQA group of query heads, only the activated experts' projections run.
4. **Attention dot product** is computed only for the active query-head experts against the (dense, shared) KV heads.
5. **Gating-weighted sum** combines the active experts' attention outputs into the layer output.

KV heads, KV cache layout, and the rest of the attention path are unchanged from GQA.

## Why it matters

- **Attention cost dominates at long context.** MoE on the MLP only helps a fixed fraction of compute; sparse activation on the attention side scales the savings as sequence length grows.
- **Right side of GQA to sparsify.** Sparsifying KV heads would shrink the cache further but break the GQA sharing structure. GQE sparsifies *only the query side*, which leaves the cache benefit intact.
- **Cheap to add on top of any GQA model.** No new positional encoding, no new normalisation — just a router on top of the existing query-head projection.

## Gotchas & tricks

- **Router collapse** is the standard MoE failure mode and applies here too. The paper uses a small auxiliary load-balancing loss; see [load-balancing-loss.md](load-balancing-loss.md).
- **k matters.** k = ⌈H_group / 2⌉ recovers half the compute saving the paper reports. Smaller k saves more but starts to hit quality; larger k is just GQA.
- **No reduction in KV cache.** GQE's win is entirely query-side FLOPs. Pair with a KV-side technique (MLA, [mla.md](mla.md)) if cache is also the bottleneck.
- **Activation-sparsity does not free up wall-clock the same way as dense compute saved.** Memory bandwidth on the dense KV side still has to be paid, so the effective speedup is workload-dependent.

## Sources

- Paper: *Grouped Query Experts: Mixture-of-Experts on GQA Self-Attention* — anonymous, 2026 — [arXiv:2606.20945](https://arxiv.org/abs/2606.20945).
- Background: *GQA: Training Generalized Multi-Query Transformer Models* — Ainslie et al., 2023.
- Background: *Mixture-of-Experts with Expert Choice Routing* — Zhou et al., 2022.
