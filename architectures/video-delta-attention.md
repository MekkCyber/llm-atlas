# Video Delta Attention (VDA)
*Depth — a frame-granular linear-memory update for video-diffusion transformers.*

**TL;DR:** Video DiTs process long spatiotemporal token sequences at every denoising step, making attention the throughput bottleneck. VDA replaces the quadratic long-range attention with a **bidirectional linear memory** that is updated **once per frame** using *all* the frame's spatial tokens jointly. Paired with local Softmax attention for fine-grained interactions and cross-modal (text/audio) heads, VDA produces the hybrid attention stack that powers **VDN-H3** — 14.3s of 768p video generated in 6.7s on 8×B200 with 8-step distillation, a 14.5× speedup over the 50-step dense H3 baseline.

**Prereqs:** [multi-head-attention](multi-head-attention.md), [../fundamentals/attention.md](../fundamentals/attention.md)
**Related:** [_linear-attention](_linear-attention.md)

---

## What it is

A **hybrid attention** design tailored for video DiTs. Two branches per attention block:

- **Local Softmax branch** — standard Softmax over a bounded local (usually intra-frame or short-window) receptive field.
- **VDA branch** — a bidirectional linear-memory recurrence over frames.

Outputs of the two branches are combined via separate output projections and learnable gates.

VDA itself is a **delta-rule-style memory** in the DeltaNet family, but with a critical design choice: memory is updated **at frame granularity** (once per frame) using all the frame's spatial tokens together, rather than token-by-token in raster order.

---

## How it works

For a video with $F$ frames each contributing $S$ spatial tokens:

```
frame f:  spatial tokens (S tokens) ─┐
                                       ├─► joint-frame update ─► M_f  (memory after frame f)
                                       │
memory M_{f-1} ────────────────────────┘
```

The joint-frame update writes all $S$ spatial tokens' keys/values into memory in one step, preserving intra-frame spatial correlations. The read is a linear query against $M_f$ for any token in the current frame.

Because the recurrence is defined per frame (not per token), the sequence-length constant that matters becomes the number of frames, not the total spatiotemporal token count.

Two more design elements:

- **Bidirectional memory.** For each frame, a forward-pass memory summarizes prior frames and a backward-pass memory summarizes future frames. Both fuse into the output — bidirectional context is available at every frame.
- **Staged teacher alignment.** Instead of retraining from scratch, VDA is grafted into a pretrained model: a teacher-alignment recipe progressively introduces the new pathway. This is what makes it deployable on top of MiniMax H3 rather than only in a from-scratch model.

The Softmax branch is preserved for text/audio conditioning heads — long-range dependencies that don't benefit from a linear approximation.

---

## Why it matters

- **Video was the last DiT frontier where dense attention was still tractable.** Frame-per-frame linear-memory attention makes livestream / interactive video generation practical: 14.5× wall-clock speedup at matched quality is the difference between offline batched rendering and interactive generation.
- **Confirms the "Softmax + linear-memory" hybrid pattern.** The pattern was already dominant in long-context LLMs (Mamba + Softmax hybrids); VDA transports it into video with the frame-granular twist.
- **Doesn't require a from-scratch model.** Teacher alignment recipe lets VDA ride into existing pretrained DiTs.

---

## Gotchas & tricks

- **Frame boundaries must be sharp.** VDA's update assumes coherent per-frame token grids; irregular masking or dynamic-frame protocols break the granularity.
- **Bidirectional memory doubles bookkeeping.** Both forward and backward passes need memory tensors — plan RAM accordingly.
- **Text/audio still needs Softmax.** VDA is not a full replacement; cross-modal heads must remain Softmax or quality collapses on prompt fidelity.
- **The staged teacher-alignment recipe is load-bearing.** Naive parameter grafting fails; the paper's staged introduction is what preserves teacher quality.
- **Serving-stack co-design.** The 14.5× number depends on 8-step distillation *and* an optimized SGLang stack; attention-side speedup alone is smaller.

---

## Sources

- Paper: *Video DeltaNet: A Video-Native Hybrid Attention for Livestream Video Generation* — Xi et al., 2026 — [arXiv:2609.20744](https://arxiv.org/abs/2609.20744).
- Family precursor: DeltaNet — Schlag et al., 2021.
- Serving stack: SGLang.
