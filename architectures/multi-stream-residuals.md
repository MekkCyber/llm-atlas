# Multi-Stream Attention Residuals (MSAR)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Instead of a single residual stream flowing between transformer blocks, MSAR maintains a small bank of parallel residual streams, each produced by a different multi-scale causal convolution over the token axis. A lightweight learned mixing layer recombines them per-token before the next attention block, giving MoE models a content-dependent way to carry multi-scale depth features. Memory stays near-baseline because only single-stream sources are stored; mixing weights are cheap. Validated at 18B / 28B MoE with a 1.58× iso-loss compute speedup over the standard residual.

**Prereqs:** [transformer-block](../architectures/transformer-block.md), [attention](../fundamentals/attention.md)
**Related:** [_moe](../architectures/_moe.md), [mla](../architectures/mla.md)

---

## What it is

A drop-in replacement for the single residual stream that connects transformer blocks. MSAR generalizes Attention Residuals (AttnRes) to multi-head *depth* attention: multiple parallel residual streams, each specialized by its own causal 1-D convolution, mixed by lightweight adaptive weights. The replacement sits inside the block boundary — attention and FFN sublayers are untouched.

## How it works

For block $\ell$ with hidden $h_\ell$:

1. **Specialize.** $K$ parallel streams are produced by causal depthwise convolutions of different kernel sizes over the token axis, each operating on the standard residual. Different kernels give each stream a different *retrieval scale* through depth.
2. **Mix.** A tiny per-token linear maps $h_\ell$ to $K$ gating weights; the next block consumes $\sum_k w_k \tilde h_\ell^{(k)}$.
3. **Store only sources.** The checkpointed state is the single-stream residual, not the $K$ streams; conv outputs are recomputed. So memory ≈ baseline.

The mixing weights make the retrieval choice *content-dependent* at every layer — effectively a lightweight depth attention.

## Why it matters

Residual streams had been treated as a fixed pipe for the full lifetime of the transformer. MSAR shows that varying the retrieval scale across depth, with learned per-token mixing, gives durable loss and downstream improvements at MoE scale. If this holds up elsewhere, it joins MLA / GQA as an architectural knob worth ablating in the next generation.

Reported gains over standard residuals, mHC, and AttnRes on 18B and 28B MoE base models. **1.58× iso-loss compute speedup** vs the standard-residual baseline — i.e. ~37% fewer FLOPs to reach the same validation loss.

## Gotchas & tricks

- Multi-scale conv kernels are small; the bulk of extra FLOPs lives in the mixing layer. Keep $K$ small (the paper uses a handful of streams) or the memory story breaks.
- The checkpointing trick (store sources, recompute streams) is what keeps activation memory close to baseline. A naive implementation that caches every stream would significantly inflate memory.
- Causal masking of the convs matters — leaking future tokens through the depth residual silently breaks autoregressive training.
- Only validated at MoE scale so far; dense-model behavior under the same recipe is open.

## Sources

- Paper: *Multi-Stream Attention Residuals* — Qin et al., 2026 — [arXiv:2610.13630](https://arxiv.org/abs/2610.13630)
