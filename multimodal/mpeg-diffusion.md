# MPEG-Diffusion
*Depth — mixed-resolution video diffusion that spends FLOPs where quality matters.*

**TL;DR:** Video diffusion models denoise every pixel at the same resolution. MPEG-Diffusion instead denoises a self-selected **high-resolution region of interest** fully and the surrounding **low-resolution backdrop** coarsely, with dual-resolution attention tying them together. Processes about 30% of full-resolution tokens for a **4.8× denoising speedup on Wan2.1-14B** at near-parity quality.

**Prereqs:** [../fundamentals/attention](../fundamentals/attention.md)
**Related:** [fastframes](fastframes.md)

---

## What it is

A mixed-resolution decoder for **DiT-style video diffusion**. Named after MPEG because it borrows the intuition that not every pixel deserves equal bits — faces and motion get priority, uniform backgrounds don't. The *selection* is learned end-to-end inside the denoiser; no external saliency model is used.

## How it works

Each denoising step splits the latent into two resolutions:

- **Hi-res crop** — the salient region, denoised at full resolution.
- **Lo-res backdrop** — the rest of the frame, down-sampled to coarse tokens.

The denoiser applies **dual-resolution attention**:

```
hi_tokens_out = Attn(hi_tokens → hi_tokens ∪ lo_tokens)   # hi attends to both
lo_tokens_out = Attn(lo_tokens → lo_tokens)               # lo stays coarse
```

The hi-res crop therefore has access to global context (via the lo-res backdrop) while the lo-res backdrop pays only for local attention among coarse tokens. A **mixed-resolution decoder** at the end stitches the two token streams back into a single full-resolution latent before the VAE.

**Region selection** is a small learned head that outputs a bbox per frame (or a soft mask over tokens), trained jointly with the denoising objective. At inference it runs once per step and is nearly free. The selector naturally tracks moving subjects across time.

## Why it matters

Video DiT compute scales quadratically with token count. Most efficiency work targets the denoiser schedule (consistency models, distillation) or the attention kernel (sparse / linear attention). MPEG-Diffusion is orthogonal: **fewer tokens processed in the first place**, and the ones that remain are the ones the output quality is sensitive to. The 4.8× denoising speedup on Wan2.1-14B is additive with distillation and quantization.

Especially strong on long videos with a mostly-static backdrop and moving subjects — exactly the regime where full-resolution is most wasteful.

## Gotchas & tricks

- **The selector must be fed soft masks during training.** Hard bbox selection is non-differentiable; the paper uses straight-through estimators / soft masks that anneal.
- **Scene cuts confuse the selector.** It relies on temporal continuity; hard cuts (editing-style) need extra training data or a reset signal.
- **Not free lunch on fully-static scenes.** If the whole frame is uniform low-detail, the selector either collapses (everything is lo-res, quality drops) or stays at default (no speedup).
- **VAE decode is NOT the bottleneck being targeted here** — that's the MPEG-Diffusion/FastFrames boundary. FastFrames attacks the VAE/KV path; MPEG-Diffusion attacks the denoiser. They stack.

## Sources

- Paper: *MPEG-Diffusion: Mixed-Resolution Video Generation* — Gu, He, V S, Song, Liu, Xie, Huang, Xu, Marculescu (Roblox / UT Austin), 2026 — [arXiv:2611.12069](https://arxiv.org/abs/2611.12069)
