# Meta Expert (cross-layer shared experts)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Classical MoE gives each layer its own private pool of experts. Meta Expert pools experts *globally* so every layer routes into the same shared library. Lower cross-layer redundancy, lower training cross-entropy, higher downstream accuracy — and at matched activation count and granularity, Meta Expert needs only **70–80 %** of the experts a vanilla MoE needs for the same quality (Tsinghua / Shanghai AI Lab, 2026).

**Prereqs:** [_moe.md](_moe.md), [transformer-block.md](transformer-block.md)
**Related:** [deepseek-moe.md](deepseek-moe.md), [capacity-factor.md](capacity-factor.md), [aux-loss-free-balancing.md](aux-loss-free-balancing.md)

---

## What it is

A cross-layer MoE variant where the expert pool is global rather than per-layer. Every MoE layer in the network routes into the same shared library of expert FFNs. The paper's claim is that layer-local MoEs waste parameters on content that recurs across depths; sharing experts across layers removes the duplication and frees capacity.

## How it works

- **Shared library.** $N$ experts live outside any single layer. Each layer's router produces top-$K$ affinities over this global library.
- **Layer-conditioned routing.** The router at layer $\ell$ is conditioned on the layer's hidden state (as usual) but may also take a layer embedding so the same expert can be used differently from different depths.
- **Residual flexibility.** Because any layer can call any expert, the hidden state has more varied residual pathways through the network; the paper argues this explains retrieval-task gains independently of raw parameter savings.
- **Balancing.** Standard balancing concerns still apply, but now balance must hold globally (across layers), not per layer — the paper's balance signals account for this.

## Why it matters

- **Memory, not FLOPs, is the MoE deployment bottleneck.** 20–30 % fewer total experts at matched activated count translates directly into a smaller served footprint — a sizeable win on frontier MoE serving.
- **Better residual pathways.** Shared experts let the hidden state revisit the same specialist from multiple depths, which the paper ties to retrieval improvements.
- **Composable with existing MoE infra.** Meta Expert is a routing-pool change, not a routing-algorithm change — it stacks with aux-loss-free balancing, fine-grained experts, and sequence-wise balance losses.

## Gotchas & tricks

- **All-to-all patterns change.** Shared experts can cause more cross-device shuffling since any token from any layer can hit any expert. Interconnect pressure may be higher than per-layer MoE at matched expert count.
- **Layer-embedding design matters.** Without layer conditioning, early layers compete with late layers for the same experts and underperform. The embedding is small but non-optional.
- **Not a free parameter cut.** The 70–80 % figure holds at matched activation count and granularity; comparing to larger coarse-grained MoEs is not apples-to-apples.
- **Checkpoint-compat concerns.** Loading a Meta-Expert checkpoint into a per-layer-MoE runtime is non-trivial; the pool indexing differs.

## Sources

- Paper: *Meta Expert: Cross-Layer Expert Sharing for Storage- and Retrieval-Efficient MoE* — Hong et al., Tsinghua · Shanghai AI Laboratory, 2026 — [arXiv:2610.27598](https://arxiv.org/abs/2610.27598).
