# FastFrames
*Depth — getting autoregressive video diffusion to interactive latency by fixing everything except the denoiser.*

**TL;DR:** Autoregressive video diffusion's per-frame latency is dominated by the VAE decode and the attention-KV update, not by the denoiser. FastFrames replaces both with lightweight alternatives and lands **36 ms/frame on a single B200 (1.3B)** and **45 ms/frame on 8× B200 (14B)** — interactive latency.

**Prereqs:** [../fundamentals/attention](../fundamentals/attention.md)
**Related:** [mpeg-diffusion](mpeg-diffusion.md)

---

## What it is

A **systems-level recipe** for AR video diffusion inference. The core model is a standard AR denoiser (per-frame diffusion conditioned on the running sequence of past frames). FastFrames' contribution is *not* the denoiser — it's that **VAE decoding and KV-cache updates eat most of the per-frame budget**, and plugging in cheaper versions of each gets you to real-time.

## How it works

Profile first. On a vanilla AR video diffusion stack, a single frame of inference breaks down roughly as:

```
denoising     ≈ 40%
VAE decode    ≈ 35%
KV update     ≈ 20%
other         ≈ 5%
```

FastFrames addresses each hot path:

**1. Lightweight VAE decoder.**
The standard SD-style VAE decoder is a stack of ResNet + attention blocks that is itself larger than the diffusion step for small denoisers. FastFrames swaps in a distilled / depth-reduced decoder that is quality-equivalent at this frame rate and ~3× cheaper.

**2. KV-cache update replacement.**
In AR video diffusion the KV cache for the condition sequence grows every frame. Naive implementations re-materialize or compress the cache every step. FastFrames uses a fixed-footprint KV representation with per-frame delta updates — a constant-time append in place of a linear-time rewrite.

**3. Scheduler tuned for the new mix.**
With VAE and KV no longer dominant, the denoiser's own schedule (step count, solver) becomes the next bottleneck. FastFrames uses a short DPM++-style schedule that was previously starved of latency budget.

Combined wins: 1.3B at 36 ms/frame on 1×B200; 14B at 45 ms/frame on an 8× B200 node.

## Why it matters

AR video diffusion is the frontier for **controllable** generation — world models, interactive video editing, in-context video tasks. The latency was the thing keeping it out of real-time products. FastFrames is the latency-engineering half of the recipe; it pairs cleanly with distillation / consistency work on the denoiser side, and is orthogonal to MPEG-Diffusion's token-count reduction.

## Gotchas & tricks

- **Lightweight VAE has to be *co-trained* with the denoiser's latent distribution.** Dropping in a generic distilled VAE degrades quality; the paper fine-tunes the VAE on the target model's latents.
- **KV delta update is model-architecture-specific.** Works out of the box for standard causal attention; needs adaptation for sliding-window or linear attention variants.
- **B200 numbers don't translate directly.** The paper's quoted latencies are Blackwell-era; on H100 the ratio of VAE to denoiser cost shifts and the speedup shrinks.
- **Interactive latency ≠ zero latency.** 36 ms/frame is ~28 fps, enough for games/world models but not for sub-10ms VR comfort budgets.

## Sources

- Paper: *FastFrames: Video Generation at Interactive Latency* — Yang, Luan, Mescheder, Dong, Richter, Koltun (Apple / Cornell), 2026 — [arXiv:2611.23226](https://arxiv.org/abs/2611.23226)
