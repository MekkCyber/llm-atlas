# Hyper-Connections
*Depth — multi-stream residual connections for deep MoE training stability; SimpleHC is the current low-overhead default.*

**TL;DR:** The transformer block normally has a single residual stream that every sub-layer reads from and writes to. Hyper-connections replace that one stream with **N parallel streams** mixed by learned matrices before each sub-layer, giving stability at depth that standard pre-norm doesn't. The 2026 **SimpleHC** reformulation strips the learned mixing matrices down to a shared, near-identity pattern, cutting overhead to **~2.4% at 30B MoE scale** while matching the original on quality and keeping the stability.

**Prereqs:** [transformer-block](transformer-block.md)
**Related:** [_normalization](_normalization.md), [../pre-training/_training-stability.md](../pre-training/_training-stability.md), [_moe](_moe.md)

---

## What it is

A modification to the residual pathway of a transformer block. Instead of one hidden state $x$ that attention and FFN both read from and update, there are $N$ hidden streams $x^{(1)}, \dots, x^{(N)}$, mixed before each sub-layer and remixed on write-back.

Motivation: in a pure pre-norm block, the residual stream's magnitude grows with depth and sub-layer contributions become negligible, which is a known cause of MoE training spikes. Multiple streams decouple the signal paths so late-layer contributions don't have to compete with a saturated residual.

## How it works

For each sub-layer $S$ (attention or FFN):

1. **Read mix.** Combine the $N$ streams into a single input $\tilde{x} = \sum_i \alpha_i \cdot x^{(i)}$ with learned (or shared) mixing coefficients.
2. **Apply sub-layer.** $y = S(\mathrm{Norm}(\tilde{x}))$.
3. **Write mix.** Distribute the update back across streams: $x^{(i)} \leftarrow x^{(i)} + \beta_i \cdot y$.

The original paper (**mHC**, 2024) made $\alpha_i, \beta_i$ learned matrices with per-sub-layer parameters, which added non-trivial parameter and memory cost at scale.

**SimpleHC** keeps the N-stream structure but:

- Removes the dense mixing matrix, replacing it with a **shared coefficient pattern** across sub-layers.
- **Shares memory access** across sub-layers so the N streams never need to be materialized separately.

The result is near-zero compute and memory overhead while the N-stream decoupling that drives the stability benefit is preserved.

## Why it matters

- Closes a stability failure class that pre-norm + QK-norm + z-loss doesn't catch: late-layer saturation on wide MoE models.
- At 15B–30B MoE, SimpleHC trains stably under deliberate initialization and data perturbations that cause baseline runs to spike.
- Before SimpleHC the overhead argument against hyper-connections was real; after it, the cost is low enough to default on.

## Gotchas & tricks

- Stream count $N$ is a hyperparameter. Small ($N=2$ to $4$) is standard; larger $N$ gives diminishing returns and multiplies KV footprint if you don't share.
- Hyper-connections stack cleanly with QK-norm and z-loss — they bound *different* drifting quantities (residual composition vs. attention/output logits).
- Debugging: if a multi-stream run diverges, one stream can go unused (its $\alpha_i$ collapses to zero). Monitor the per-stream norms, not just the fused one.

## Sources

- Paper: *On the Design of Hyper-Connections: Simplification, Stability, and Scalability* — Qin et al., 2026 — https://arxiv.org/abs/2611.05984
- Paper: *Hyper-Connections* (mHC original) — 2024.
