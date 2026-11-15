# Hyper-Connections (SimpleHC)
*Depth — generalized residual pathway for transformer blocks, in its simplified "SimpleHC" form.*

**TL;DR:** Hyper-Connections (HC) are a family of generalized residual-stream designs for transformer blocks that improve training stability over vanilla residual connections. Earlier HC variants added nontrivial compute by mixing multiple residual lanes. **SimpleHC** shows the mixing isn't load-bearing: drop residual mixing and share one memory read/write across sublayers, and you recover the full HC benefit at ~2.4% extra wall time over a vanilla residual baseline. Matches the strongest prior HC variant (mHC) on 15B and 30B MoE training runs and improves stability under stress tests.

**Prereqs:** [transformer-block](transformer-block.md), [_normalization](_normalization.md)
**Related:** [../pre-training/_training-stability](../pre-training/_training-stability.md), [_moe](_moe.md)

---

## What it is

A transformer block's residual connections are what let gradients flow and activations stay bounded. Hyper-Connections generalize the single additive residual into a *small bank of residual lanes* that interact through learned mixing — stabilizing the stream at scale but adding compute and memory per block.

SimpleHC is the minimal HC: keep the bank, **drop the mixing**, and share a single memory access (read + write) across the sublayers of a block. The net effect is a reshape of how residuals flow with nearly no overhead.

## How it works

- Each block maintains n parallel residual lanes instead of one.
- Sublayers (attention, MLP, MoE) each write to one lane and read from the others; the "shared memory access" means this read/write pattern is amortized across sublayers rather than repeated.
- **No learned mixing matrix** between lanes — removing it is what makes SimpleHC nearly free.
- The lanes are summed at block exit to produce the output used by the next block.

The paper verifies with ablations that the residual-stream statistics SimpleHC produces are very close to those of mHC (the strongest prior HC variant), and that the removed mixing is responsible for most of mHC's compute overhead without contributing much to its benefit.

## Why it matters

- **Near-free stability.** 2.4% end-to-end training overhead over vanilla residual at 30B MoE scale — cheap enough to turn on by default.
- **Scales to MoE.** Validated at 15B and 30B dense-active parameter MoE models, which is where residual-stream instabilities bite hardest.
- **Replaces mHC.** For teams already running the mixed variant, SimpleHC is a direct simplification with matched quality.

## Gotchas & tricks

- The "shared memory access" is an implementation detail: sublayers share the lane-read/write schedule within a block. Naively re-implementing HC with per-sublayer reads reintroduces the overhead.
- Benefit on dense (non-MoE) smaller models is smaller; the paper argues HC-family tricks pay off most where residual-stream width and sparsity make vanilla residuals wobble.
- Not a drop-in for pre-norm vs post-norm choice — SimpleHC composes with either, but normalization placement still matters independently.

## Sources

- Paper: *On the Design of Hyper-Connections: Simplification, Stability, and Scalability* — Qin et al., SJTU / Xiaohongshu / PKU / NTU, 2026 — arXiv 2611.05984 — https://arxiv.org/abs/2611.05984
