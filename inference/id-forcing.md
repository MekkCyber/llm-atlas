# In-Distribution Forcing (ID-Forcing) for Long-Video Generation
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Autoregressive video diffusion drifts past the training horizon because cached KV entries themselves become out-of-distribution during rollout (the **KV-provenance problem**). ID-Forcing (2026) fixes it at test time with **self-caching**: each chunk is cached *without attending to prior KV entries*, so the rolling KV window matches the training-time distribution exactly. Extends short-horizon AR video diffusion to minute-scale generation.

**Prereqs:** [../fundamentals/attention.md](../fundamentals/attention.md)
**Related:** [../architectures/multi-head-attention.md](../architectures/multi-head-attention.md) · [../architectures/dynamic-sparse-attention.md](../architectures/dynamic-sparse-attention.md)

---

## What it is

AR video diffusion models generate chunks of video conditioned on earlier chunks via KV conditioning: cached key/value tensors from the past constrain the next chunk. The convention is to compute each new chunk's KVs *with* attention over previous chunks' KVs — the same way an AR text LM extends its context.

Beyond the training horizon that conditional construction becomes a liability. The cached KVs are built from distributions the model never trained on, so **the KV entries themselves are OOD**. KV conditioning (selecting or modifying which KV entries attend to what) doesn't fix this: it assumes the entries are in-distribution. Drifting (shifting colors, decaying motion) is the symptom.

## How it works

- **Self-caching.** For each chunk, build the chunk's cached KVs *without attention to prior KV entries*. The chunk is still conditioned on prior output via the usual denoising context, but the KV tensors it stores are computed as if the chunk were an independent short clip — exactly as during training.
- **Rolling window.** As generation proceeds, old chunks' self-cached KVs scroll out of the window and new ones scroll in; the window's distribution is always training-matched.
- **No training changes.** ID-Forcing is a **test-time** method applied to any pretrained AR video diffusion backbone.

## Why it matters

- Shifts the long-video problem from "fix the policy" to **"guard the cache"**: a cheap, training-free intervention composes with any AR video diffusion backbone.
- Enables **minute-scale generation** with short-horizon models — competitive on standard benchmarks and substantially better by drift metrics / user study.
- Frames a reusable principle: **make test-time cache construction match training-time distribution**. The same idea could apply to any auto-regressive model where the cache is built differently at train and inference time.

## Gotchas & tricks

- **Loses some context propagation.** Self-caching sacrifices the model's ability to carry fine detail across chunks via KV attention. The paper trades that for stability; some applications may want a hybrid.
- **Drift metrics matter.** Standard quality benchmarks on short clips don't catch drift; use dedicated drift metrics or user studies for long videos.
- **Composition.** ID-Forcing is orthogonal to attention-level sparsification (e.g., [../architectures/dynamic-sparse-attention.md](../architectures/dynamic-sparse-attention.md)) and to RLHF-style post-training for video — stack them.
- **Not for text LMs as-is.** Text LMs' KVs are already built the same way at train and inference, so the KV-provenance problem doesn't apply directly.

## Sources

- Paper: *In-Distribution Forcing for Long Video Generation at Test Time* — 2026 — [arXiv:2610.03120](https://arxiv.org/abs/2610.03120).
