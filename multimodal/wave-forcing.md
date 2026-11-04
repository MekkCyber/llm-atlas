# Wave Forcing
*Depth — streaming video diffusion with a wavefront denoising schedule across (time × step), pipelined over multiple GPUs.*

**TL;DR:** Rolling-diffusion video generators denoise frame-chunks sequentially; throughput is limited by per-chunk sequential steps. **Wave Forcing** (SJTU DENG Lab, 2026) organizes compute along a *wavefront* in the (time × denoising-step) grid: at any instant, early frames are near-clean while later frames are still noisy, and each GPU works on one diagonal of the wavefront. Combined with a three-stage training recipe that teaches the model to tolerate this causal shape and a runtime that pipelines denoising + KV-cache refresh, a 4-step model hits 134.8 end-to-end FPS on 8× H200 GPUs.

**Prereqs:** [../architectures/multi-head-attention.md](../architectures/multi-head-attention.md)
**Related:** [../systems/dualpipe.md](../systems/dualpipe.md) · [history-forcing.md](./history-forcing.md)

---

## What it is

Streaming video diffusion generates frames in real time, chunk by chunk, as they are consumed by a downstream renderer. Three constraints fight each other:
- Each frame needs `k` denoising steps to be clean.
- Frames must be released in order.
- Latency per frame must stay below the playback-rate budget.

**Rolling Forcing** (prior art) denoises chunk-`i` to completion, then moves to chunk-`i+1`, with KV-cache refresh at chunk boundaries. Sequential across chunks.

**Wave Forcing** instead shears compute across (time × step). Think of a 2-D grid: rows = time chunks (`i = 1, 2, …`), columns = denoising steps (`s = 1, 2, …, k`). A **wavefront** is a diagonal of this grid: at wall-clock time `τ`, chunk `i` is at step `s_i(τ) = max(0, τ − i + 1)` (bounded by `k`). Early-in-timeline chunks are advanced; later chunks are starting fresh.

Per wavefront advance, each GPU owns one or a few diagonal cells, runs that cell's denoising matmul, exchanges KV updates with its neighbours, and advances the wavefront by one step.

## How it works

1. **Causal wavefront schedule.** The model's attention over time must respect the fact that, at a given wavefront position, late-timeline frames are *noisier* than early ones. The training recipe explicitly includes this asymmetry — the model learns to attend across a noise-level-staggered context.
2. **Three-stage training.**
   - Stage 1: standard diffusion training on video clips.
   - Stage 2: fine-tuning on wavefront-shaped noise schedules.
   - Stage 3: few-step distillation targeting 4–5 steps per frame.
3. **Wave Runtime.** Across `G` GPUs, the wavefront is partitioned so each GPU holds a contiguous slice of time-chunks. Denoising steps and KV-cache refreshes pipeline across GPUs; the runtime hides communication under compute by overlapping the next-step matmul with the previous step's all-reduce.

## Why it matters

- **Real-time video generation.** 134.8 FPS end-to-end on 8× H200 takes video diffusion past interactive thresholds — relevant for games, streaming content, and video-based world models used by agents.
- **Scheduling is first-class.** Demonstrates that *how* diffusion compute is laid out across (step × time × GPU) is as important as the model architecture. The same model under naive scheduling would be far slower.
- **Compositional with model advances.** The scheduling trick is model-agnostic; architecture improvements stack on top.

## Gotchas & tricks

- **Quality vs step count.** 4-step Wave Forcing matches Rolling Forcing on motion but is weaker on fine detail; 5-step closes the gap. Pick step count by the downstream use case.
- **KV cache pressure.** Many-GPU wavefront needs enough on-device KV capacity to hold several chunks' worth of context. OOMs show up at scale-up, not scale-down.
- **Attention shape is noise-staggered.** Attention must support mixed-precision noise levels within one tile; naive Flash Attention won't do.
- **End-to-end FPS ≠ model FPS.** The published number is end-to-end including decode and transport; model-only FPS is higher but less meaningful.

## Sources

- Paper: *Wave Forcing: Towards Speed-of-Light Streaming Video Generation* — Jin et al., SJTU DENG Lab / PKU / XJTU / ECNU / NTHU, 2026 — [arXiv:2611.00859](https://arxiv.org/abs/2611.00859).
- Paper: *Rolling Forcing* — prior art for streaming video diffusion referenced by the authors.
