# Block Attention Residuals (AttnRes)
*Depth — cross-block residuals that route completed anchor-layer outputs into later linear-attention layers.*

**TL;DR:** In a hybrid linear/softmax stack, only the periodic softmax anchors produce full-rank token interactions; downstream linear layers see progressively rank-collapsed features. **Block Attention Residuals** carry each completed anchor block's summary forward as a residual into later linear layers, giving them a fresh full-rank signal to compose with instead of relying only on the (rank-limited) linear stream. Reported to lift deep-layer effective rank by ~12% and to be the key architectural addition that makes the hybrid competitive with pure softmax at long sequences.

**Prereqs:** [hybrid-linear-softmax-attention](hybrid-linear-softmax-attention.md), [transformer-block](transformer-block.md)
**Related:** [multi-head-attention](multi-head-attention.md)

---

## What it is

A cross-layer routing pattern applied on top of hybrid linear/softmax stacks. For each softmax anchor block, its output activations are stashed and re-injected as an additive residual into one or more later linear-attention layers. Unlike ordinary skip connections (which run only within a block or one layer forward), AttnRes routes *across many layers* and specifically from *anchor* blocks into *linear* blocks.

## How it works

Given a stack of blocks $B_1, B_2, \dots, B_L$ where a subset $\mathcal{A} \subset \{1,\dots,L\}$ are softmax anchors, ordinary residual flow gives each block

$$h_{i} = h_{i-1} + f_i(h_{i-1}).$$

AttnRes augments the pre-block hidden state of a *linear* block $i$ with the summary of the *most recent anchor block* $a(i) \in \mathcal{A}$, $a(i) < i$:

$$h_{i} = h_{i-1} + f_i\!\big( h_{i-1} + \alpha_i \cdot g(h_{a(i)}) \big),$$

where $g$ is a lightweight projection (frequently identity or a linear map into the block's residual space) and $\alpha_i$ is a learned scalar. Downstream linear layers therefore compose over both their nearest neighbour's output *and* the last softmax-refreshed representation.

Effect: the deep-layer effective rank of the token-mixing operator stays high across the hybrid stack — the paper reports ~12% higher deep-layer effective rank vs the plain hybrid.

## Why it matters

Hybrid stacks solve the *how often* problem of restoring full-rank mixing; AttnRes solves the *how far* problem. Without it, only the linear layers immediately following an anchor benefit from the refresh — later layers drift back toward rank collapse. AttnRes broadcasts the refresh forward, letting a small softmax budget do more work.

## Gotchas & tricks

- **Choose one anchor source per linear layer.** Summing many anchor blocks into one residual doubles memory traffic without clear quality gains; the "most recent anchor" default performs best in the paper's ablations.
- **Learned scaling matters.** A fixed $\alpha = 1$ can destabilize training; learned per-layer $\alpha_i$ is standard practice.
- **Pair with periodic anchor placement.** AttnRes assumes a fresh anchor is at most a few layers back. Clustered anchor placements (all early, all late) make the residual go stale.
- **Trivially compatible with existing kernels.** No changes to attention kernels themselves; only the residual wiring changes.

## Sources

- Paper: *SANA-Video 2.0: Hybrid Linear Attention with Attention Residuals for Efficient Video Generation* — Chen et al., NVIDIA, 2026 — [arXiv:2607.21553](https://arxiv.org/abs/2607.21553).
