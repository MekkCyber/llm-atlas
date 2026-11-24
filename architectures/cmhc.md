# cmHC — Contracted Manifold Hyper-Connections
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A parameter-free fix to [manifold-constrained Hyper-Connections](hyper-connections.md) (mHC). The mHC mixing matrix lives on the Birkhoff polytope of doubly-stochastic matrices; at its boundary, streams decouple (under-mixing), while over-mixed shallow layers blur them. cmHC **contracts the polytope toward its interior** with a single learned scalar per layer, with heavier contraction at shallow depth. No new parameters, and fused Ascend NPU kernels cut hyper-connection op latency by up to 44.8%.

**Prereqs:** [hyper-connections](hyper-connections.md), [transformer-block](transformer-block.md)
**Related:** [_moe](_moe.md) · [deepseek-moe](deepseek-moe.md)

---

## What it is

mHC keeps the per-layer mixing matrix `P` doubly stochastic — rows and columns sum to 1 — i.e. on the Birkhoff polytope. The polytope's vertices are permutation matrices (streams swap, no actual mixing) and its interior point `J/n` (the all-`1/n` matrix) is maximal mixing. cmHC shows mHC under-mixes at the boundary and over-mixes at the shallow-layer interior, and replaces `P` with a convex combination:

```
P' = c · P + (1 − c) · (J / n)
```

where `c ∈ [0, 1]` is a learned scalar per layer, trained end-to-end. Lower `c` means more contraction toward the uniform-mixing matrix.

---

## How it works

**Depth-aware contraction schedule.** Shallow layers suffer the over-mixing failure mode; cmHC initializes `c` smaller at shallow depth (more contraction) and larger at deeper layers, letting the model recover the full polytope only where it needs the expressiveness. The schedule is a function of layer index set at init and refined by training.

**Why interior-only contraction works.** The polytope boundary contains "near-permutation" matrices that pass one stream through almost unchanged. Deep stacks of these decouple streams catastrophically. Pulling `P` toward `J/n` injects a floor of mixing without killing the polytope structure. Because `J/n` is itself doubly stochastic, `P'` stays on the polytope — the constraint is preserved automatically.

**No new parameters.** The only addition is the scalar `c` per layer. Everything else — the Sinkhorn-style projection enforcing double stochasticity, the read/write projections — is unchanged from mHC.

**Fused op.** On Ascend NPUs the authors ship a fused kernel for the whole hyper-connection block (read / mix / write), reducing op latency by up to 44.8% vs the unfused mHC baseline.

---

## Why it matters

Hyper-Connections are an increasingly-standard residual-stream upgrade in open MoE stacks, but their failure modes were poorly characterized. cmHC gives a principled, zero-parameter fix and a deployment-ready kernel. It also moves the "add more streams" conversation to "shape the mixing surface" — a more productive design axis for residual-stream research.

---

## Gotchas & tricks

- **Initialize `c` conservatively.** The paper's schedule starts small at shallow layers and grows with depth. Uniform init tends to recover mHC behavior.
- **MoE interactions:** reported gains include MoE training stability — hyper-connection mixing interacts with expert routing variance, and contraction appears to reduce the pathology.
- **Sinkhorn step still needed** — `P'` stays on the polytope, but `P` before contraction is still produced by the usual projection; don't skip it.

---

## Sources

- Paper: *cmHC: Balancing Stream Mixing in Hyper-Connections via Geometric Contraction* — He et al., 2026 — [arXiv:2611.27193](https://arxiv.org/abs/2611.27193)
