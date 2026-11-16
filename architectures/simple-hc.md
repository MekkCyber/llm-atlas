# SimpleHC
*Depth — a cost-reduced variant of Hyper-Connections.*

**TL;DR:** Hyper-Connections (HC) replace the single residual stream with several parallel connection paths, trading more parameters and compute for better gradient stability at scale. SimpleHC strips HC to its cheap core: drop the learned residual-mixing matrix, and share the hyper-connection buffer across sublayers inside a block. Matches multi-HC quality on 15B–30B MoE while adding only ~2.4% training time over a vanilla residual.

**Prereqs:** [transformer-block.md](transformer-block.md)
**Related:** [_normalization.md](_normalization.md), [qk-norm.md](qk-norm.md)

---

## What it is

A drop-in replacement for the residual stream in a transformer block. In the original Hyper-Connections work, each sublayer (attention, MLP) reads from and writes to a *set* of n parallel streams through a learned n×n mixing matrix. The matrix stabilizes gradient flow but costs parameters and all-to-all memory traffic. SimpleHC asks which parts of that mechanism actually do the stabilization work and keeps only those.

## How it works

Two simplifications against multi-HC:

1. **No residual-mixing matrix.** Sublayers concatenate the n incoming streams and apply a plain projection. Mixing weights are learned implicitly inside the projection rather than through an explicit n×n matrix per-step.
2. **Shared memory across sublayers.** One hyper-connection buffer is shared by all sublayers in a block. Each sublayer has its own read/write heads into that buffer, but the buffer itself is not duplicated per sublayer. Parameter cost and memory-bandwidth cost scale with the number of heads, not with sublayer count.

Effectively: keep the "n parallel streams" shape (it is what makes gradient flow robust), kill the per-step mixing matrix (the expensive part), and share the physical buffer (the memory-bandwidth part).

## Why it matters

Hyper-connections have been a known stability trick but were too costly for routine use at the MoE frontier. SimpleHC's ~2.4% training-time overhead at 30B means it is cheap enough to default on, which matters as MoE depth and expert count keep growing — those are exactly the regimes where residual-stream gradient issues bite hardest.

## Gotchas & tricks

- The "share buffer across sublayers" trick is where most of the cost reduction comes from — if you re-introduce per-sublayer buffers, compute and memory traffic blow back up.
- The paper reports SimpleHC on MoE; pure dense behavior at that scale is not shown.
- Hyper-connections interact with normalization placement — like QK-norm, gains compound rather than swap.
- Head count (how many read/write heads into the shared buffer) is the knob that trades stability vs. cost. Too few heads collapse back toward a plain residual; too many re-inflate memory traffic.
- When migrating an existing checkpoint, initialize the shared buffer so the sum of read heads equals the old residual — that way training resumes without a loss spike.

## Sources

- Paper: *On the Design of Hyper-Connections: Simplification, Stability, and Scalability* — Qin, Zou, Dai, Shi, Xia, Yang, Chen, Liu, Cheng, Yan, 2026 — [arXiv:2611.05984](https://arxiv.org/abs/2611.05984)
