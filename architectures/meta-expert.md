# Meta Expert
*Depth — cross-layer expert sharing for storage-efficient MoE.*

**TL;DR:** In mainstream MoE models, each transformer layer owns a private pool of experts and the same semantic content is often re-learned across layers. Meta Expert replaces per-layer private pools with a single shared pool of experts that every layer can route to. Per-layer routers decide which shared experts to invoke, which collapses the total expert parameter count while giving hidden states more flexible residual pathways across depth.

**Prereqs:** [_moe.md](../architectures/_moe.md), [deepseek-moe.md](../architectures/deepseek-moe.md)
**Related:** [aux-loss-free-balancing.md](../architectures/aux-loss-free-balancing.md), [transformer-block.md](../architectures/transformer-block.md)

---

## What it is

MoE pools so far have been *layer-local*: layer `L` has its own `N_L` experts, no sharing across depth. Meta Expert makes the expert pool *global* — one pool of shared "meta experts" that every layer can route to. Routing stays per-layer: each layer has its own router that selects top-k meta experts from the shared pool for every token.

## How it works

- A single shared bank of `E` experts replaces the per-layer `L · E` private banks.
- Each transformer layer `l` has its own router `R_l` that scores all `E` shared experts and selects top-k for the current token.
- Expert outputs are mixed into the residual stream of layer `l` as in standard MoE; the next layer's router independently selects from the same shared pool.
- Load balancing uses an aux-loss-free bias scheme adapted to the shared setting (per-expert utilization aggregated across layers, not per-layer).

Hidden states can revisit the same expert at different depths — a form of structural weight tying that increases the number of distinct residual-pathway compositions without adding parameters.

## Why it matters

- **Storage/HBM.** Expert weights dominate the on-disk and in-HBM footprint of trillion-scale MoE models. Shared experts cut the parameter count roughly by the number of layers that share them.
- **Retrieval accuracy.** The paper reports improved information-retrieval probing: giving hidden states more residual-pathway options makes it easier to recover content stored earlier in the stack.
- **Matches Vanilla MoE accuracy at ~70–80% of experts**; at the same expert budget, lower loss and higher downstream accuracy.

Trade-offs: shared experts must generalize across layer-level roles (early-layer feature extraction vs. late-layer composition), so under-provisioned shared pools can underperform; the paper's regime provisions enough capacity to avoid this.

## Gotchas & tricks

- Load balancing *across layers*, not within one layer, matters: naive per-layer balance loss can starve some layers of a popular expert. Aggregate utilization globally.
- Routing redundancy — if several layers consistently route to the same expert, the pool can effectively collapse. Monitor per-layer router entropy.
- Mixing shared Meta Experts with a small set of per-layer private experts (hybrid) is a natural follow-up but not evaluated in the paper.

## Sources

- Paper: *Meta Expert: Cross-Layer Expert Sharing for Storage- and Retrieval-Efficient MoE* — Hong et al., 2026 — [arXiv:2610.27598](https://arxiv.org/abs/2610.27598)
