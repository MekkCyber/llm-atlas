# Self Gradient Forcing (SGF)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Autoregressive video diffusion trained with Self Forcing feeds the student its own rollout KV cache to fight exposure bias, but that cache is frozen during backprop — future-frame losses can't teach the model how to *write* useful memory. SGF is a two-pass training strategy that reintroduces gradient flow into the historical KV cache without backpropagating through the full serial rollout. With a 5-second training window, it extrapolates to multi-minute video.

**Prereqs:** [../fundamentals/attention.md](../fundamentals/attention.md), [_training-stability.md](./_training-stability.md)
**Related:** [mtp.md](./mtp.md), [../inference/README.md](../inference/README.md)

---

## What it is

**Self Forcing** (the baseline SGF builds on) trains AR video diffusion by rolling the student out and using its own generated latents as historical context — mirroring inference conditions and reducing exposure bias. The historical KV cache is treated as fixed rollout state and passed forward. Backprop happens on the future losses, but never flows *into* the cached representations, so the model has no gradient signal telling it how to write context that will be useful downstream. The paper calls this the **historical context-gradient gap**.

## How it works

SGF is a **two-pass** training procedure that reconstructs the missing gradient signal without paying for full-rollout backprop.

**Pass 1 — no-grad rollout.** Run the autoregressive rollout in inference mode (no gradients tracked). At a randomly sampled denoising exit step, record: (a) the self-generated context latents produced up to that point, and (b) the noisy latents that would be fed to the model at that step.

**Pass 2 — parallel context-gradient reconstruction.** Feed the model the recorded context latents as **stop-gradient clean-latent inputs** and the noisy latents from Pass 1. The model recomputes the context KV representations and the future-to-context causal attention — this time with gradients tracked. The future-frame loss now flows *into* the KV representations of the historical context, teaching the model to encode context into more effective causal memory.

The stop-gradient on the context latents themselves is critical: it prevents the loss from trying to change what was generated, only how it gets represented in the cache.

## Why it matters

Long-horizon consistency is the hardest failure mode of AR video diffusion — subjects morph, backgrounds drift, layouts break. SGF is a *training-side* fix that doesn't require a bigger context window, an external memory module, or a bigger model. It's compatible with existing Self Forcing pipelines and preserves the native autoregressive inference path.

Empirically: a model trained on 5-second windows can extrapolate to several-minute videos with better subject identity, background/layout consistency, and temporal stability than Self Forcing alone.

## Gotchas & tricks

- Cost per training step is roughly 2× a normal forward, since Pass 2 recomputes attention. Cheaper than backpropagating through the full rollout.
- The sampled exit step controls the horizon at which gradient reconstruction happens — too shallow and you supervise mostly early context; too deep and you're near the tail of the rollout with weak signal. Uniformly sampling exit steps during training is the default.
- Stop-gradient placement matters. If you accidentally let gradients flow through the generated context latents, you're re-introducing full-rollout backprop and the memory-writing signal gets tangled with the denoising signal.

## Sources

- Paper: *Self Gradient Forcing: Native Long Video Extrapolation* — Zhuang et al., 2026 — [arXiv:2607.20368](https://arxiv.org/abs/2607.20368)
- Baseline: *Self Forcing* — the AR-video-diffusion training regime SGF extends.
