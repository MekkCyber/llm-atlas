# Streaming Video Diffusion
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A **four-step causal** diffusion / flow video model trained with **Progressive Forcing** — a curriculum that gradually weans the model off teacher-forced ground-truth frames during training. Combined with a proxy-conditioned renderer (Invoker-Render), produces stable interactive video rollouts of 30+ minutes, beating recent streaming video baselines.

**Prereqs:** [_flow-matching.md](_flow-matching.md)
**Related:** [decoder-free-sampling.md](decoder-free-sampling.md), [README.md](README.md)

---

## What it is

Interactive / infinite video generation ("world model" demos, playable videos, long game rollouts) needs a model that (a) decodes frames fast — a handful of denoising steps, and (b) remains stable over long horizons without drifting into mode collapse or visual chaos.

Standard video diffusion models are bidirectional over a fixed clip length. Streaming video needs *causal* generation (frame $t$ depends only on frames $< t$) and *few-step* sampling. Training such a causal few-step model naively is unstable: the model is both learning to denoise and to handle its own accumulating errors.

Progressive Forcing introduces a curriculum: early in training, feed mostly teacher-forced (ground-truth) previous frames; gradually swap in the model's own predicted previous frames, until training matches the deployment distribution.

---

## How it works

1. **Architecture.** A causal streaming transformer / diffusion backbone emits each frame conditioned on previous frames (model-predicted at inference, mixed at training). Four-step sampling per frame.
2. **Proxy conditioning.** The renderer (**Invoker-Render**) accepts structural proxies (skeletons, flow maps) + an appearance reference + text. A separate *bidirectional creation model* produces the proxies.
3. **Progressive Forcing training curriculum.** At step $k$, the probability of using the model's own previous frame (vs ground truth) is $p(k)$, annealed from 0 to 1 over training. Early training is essentially teacher-forced denoising; late training matches inference.
4. **Stable long rollouts.** The authors report 30-minute racing-game rollouts without divergence.

---

## Why it matters

- **Interactive video gen becomes practical.** 4-step causal sampling + stable rollout is the combination needed for Genie-like playable worlds and infinite-horizon video demos.
- **Progressive Forcing is reusable.** The curriculum pattern applies to any autoregressive-with-own-outputs training (streaming audio, long-document generation) where distribution shift between training and inference is the failure mode.
- **Separating proxy generation from rendering.** The two-model split makes the rendering problem independently tractable and the proxy problem controllable.

---

## Gotchas & tricks

- **Curriculum schedule matters.** Too fast → model never fully learns to denoise; too slow → wastes training on in-distribution. The paper tunes $p(k)$ per model.
- **Four steps requires rectification or distillation.** A base flow model usually needs rectified-flow training or distillation to sample well at 4 steps.
- **Proxy quality bounds output quality.** If the proxy model drifts, Invoker-Render faithfully renders drifted proxies. Stability of the proxy model is the long-rollout bottleneck, not the renderer.
- **Not latent-free by default.** Compose with [decoder-free-sampling](decoder-free-sampling.md) if you want to also skip the VAE decode.

---

## Sources

- Paper: *Code World Model 2.0: Towards Infinite Controllable Video Generation* — Wang, Zhong, Yang, Tian, Liu, Gong, Lu, Tian, Shen, Lan, Zhang (Westlake / Inclusion AI / Zhejiang) — 2026 — [arXiv:2611.22925](https://arxiv.org/abs/2611.22925).
- Background: *Rectified Flow* — Liu et al., 2022 — few-step sampling precondition.
- Related: Genie-family interactive world models.
