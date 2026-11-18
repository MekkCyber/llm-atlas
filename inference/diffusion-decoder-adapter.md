# Diffusion Decoder Adapter (Uno)
*Depth — a plug-in diffusion head that accelerates autoregressive decoding with no separate draft model.*

**TL;DR:** A small diffusion adapter attached to the top of a pretrained autoregressive LLM that predicts several future tokens at once by denoising. Because it reads the base model's own hidden states, it functions as a **self-speculator** — a drafter that is literally part of the target model — and a standard speculative-decoding verify step (the base model's next-token distribution) decides what to accept. K2-Horizon's **Uno** adapter is lossless: accepted draft tokens are provably samples from the target distribution, with reported **up to 2.2×** end-to-end decoding speedup across the K2-Horizon fleet without a separately-trained draft model.

**Prereqs:** [_speculative-decoding.md](_speculative-decoding.md), [../pre-training/mtp.md](../pre-training/mtp.md)
**Related:** [loopfill.md](loopfill.md)

---

## What it is

A lightweight diffusion head trained on top of a frozen (or lightly fine-tuned) autoregressive base model. Given the base model's hidden states at the current position, the adapter runs a few denoising steps to produce *several* future tokens in parallel. Those tokens are then fed back into the base model as a draft; the base model's forward pass at those positions verifies the draft under the usual speculative-sampling acceptance rule (Leviathan 2023), so the final distribution is exactly the target's.

Unlike EAGLE/Medusa, the adapter is a *diffusion* model, not an autoregressive drafter. Unlike a separate small draft model, it reuses the base model's hidden states and lives in the same checkpoint.

## How it works

Three pieces:

1. **Base signal.** At each decode position *t*, take the base model's top-layer hidden state `h_t^L` and feed it as conditioning to the diffusion adapter.
2. **Parallel denoising.** The adapter maintains a small latent block spanning positions `t+1..t+k`. A few denoising steps (typically 2–4) iteratively refine that block into token ids `ŷ_{t+1..t+k}`. Fewer denoising steps = cheaper draft; more = higher acceptance.
3. **Speculative verify.** The base model is then asked for its next-token distribution at `t+1..t+k` (one parallel pass). Standard speculative-sampling acceptance (Leviathan) keeps the longest prefix whose per-step distribution the base agrees with. Non-accepted draft tokens are resampled from the base's correction distribution. The output is *exactly* distributed as plain autoregressive decoding.

Because the draft is a diffusion head, not an AR model, it can propose multiple tokens that depend on each other in both directions — closer to a lattice than a chain — which pushes acceptance rates up on repeated or structured text.

## Why it matters

- **No separate draft checkpoint.** EAGLE needs a trained draft head; Medusa needs extra target heads. Uno's adapter ships with the model.
- **Lossless.** Not an approximation: the final sampling distribution is the target's.
- **Decouples quality and draft cost.** Denoising step count is a knob at serving time, not a retraining decision. Serving configurations can trade draft cost for acceptance rate on the fly.
- **Up to 2.2× speedup** across the K2-Horizon fleet, per the release.

## Gotchas & tricks

- The adapter must see training data sampled from the target's own distribution, or acceptance collapses. Train it after the base model is frozen; distilling it from a different model is not the same.
- Diffusion step count has a real sweet spot: too few and acceptance is low; too many and the draft cost eats the win. K2's defaults are small (2–4 steps) because tokens are discrete and the adapter is small.
- Combining with KV-cache quantization is fine for the verify pass, but the adapter's hidden-state input may need higher precision than FP4 to keep acceptance stable.
- On looped LLMs the adapter is still useful but the dual-axis technique (see [loopfill](loopfill.md)) is a better fit — Uno is designed for standard autoregressive targets.

## Sources

- Paper: *K2-Horizon: A Frontier Fleet of Open Source Language Models* — K2 team, MBZUAI, 2026 — [arXiv:2611.21003](https://arxiv.org/abs/2611.21003) — introduces the Uno diffusion decoder adapter.
- Context: *Fast Inference from Transformers via Speculative Decoding* — Leviathan et al., 2023 — [arXiv:2211.17192](https://arxiv.org/abs/2211.17192) — the acceptance rule Uno uses.
