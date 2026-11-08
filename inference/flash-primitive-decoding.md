# Flash primitive decoding
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A non-autoregressive decoding head for visual planning: predict the full set of coordinates for points, bounding boxes, or trajectories in **one forward pass**, instead of emitting each digit as a token. Introduced in VFlash (NVIDIA, 2026) for VLM-based robot planning. Collapses latency ~25× for long 3D trajectories while *improving* accuracy over autoregressive emission.

**Prereqs:** [attention](../fundamentals/attention.md), [multi-head-attention](../architectures/multi-head-attention.md)
**Related:** [multi-head-attention](../architectures/multi-head-attention.md)

---

## What it is

When a VLM is used as a planner, outputs like "reach to (x, y, z)" or a 20-step trajectory are structurally short — a handful of numbers — but current VLMs emit them one digit-token at a time via standard autoregressive decoding. A 3D trajectory of 50 waypoints can be hundreds of tokens. Flash primitive decoding replaces the autoregressive tail with a parallel head that predicts the whole numeric structure in one step.

## How it works

A small "drafter" module reads the VLM's hidden state at the "primitive start" token and emits a fixed-shape tensor of coordinates directly, bypassing the LM head. Per primitive type (points, boxes, trajectories), the head has a different output shape but the same single-pass mechanism. The main LM still produces grounding, reasoning, and natural-language parts of the response token-by-token; only the numeric primitive is non-autoregressive.

At training time, the primitive head is supervised on ground-truth coordinates via regression (MSE / smooth-L1); language tokens keep standard cross-entropy. The drafter is lightweight — a few MLP/transformer layers over the primitive start hidden state.

## Why it matters

Robot/VLA stacks hit a wall at interactive rates because token-by-token coordinate emission bottlenecks both latency and accuracy (compound errors across digits). Non-AR structured heads are the obvious fix — the paper is one of the first demonstrations at VLM scale, with a general recipe across primitive types.

The approach generalises beyond robotics: any domain where VLM outputs are structured numeric data (chart values, bounding boxes for OCR, UI widget coordinates) should see similar wins.

## Gotchas & tricks

- The drafter head is **primitive-type conditional**. Mixing primitive types in one response requires a token to switch head mode (either explicit control tokens or learned type prediction).
- Error behaviour is different from AR: AR errors propagate left-to-right (often making one digit wrong while the rest is correct); non-AR errors are **jointly** off, which tends to be easier to detect but harder to partially recover.
- The drafter must not reduce the planning latents the main VLM uses for the rest of the response — ablations show detaching the drafter's gradient from the backbone helps avoid regressions on natural-language quality.

## Sources

- Paper: VFlash: Fast, Accurate Visual Planning via Flash Primitive Decoding — Wang et al., 2026 — [arXiv:2611.08449](https://arxiv.org/abs/2611.08449)
