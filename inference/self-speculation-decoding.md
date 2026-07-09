# Self-speculation decoding
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Use the *same model weights* as both the drafter and the verifier by exploiting two different decoding modes trained into one checkpoint. In Nemotron-Labs-Diffusion, the model is trained with a **joint AR + diffusion objective** and at inference can operate in AR, diffusion, or self-speculation mode: **diffusion drafts a block of tokens in one forward pass, then AR verifies them sequentially** — all inside the same weights. No separate drafter to train or align; the drafter–verifier distribution gap that plagues classical speculative decoding disappears by construction.

**Prereqs:** [../pre-training/mtp.md](../pre-training/mtp.md), [../pre-training/joint-ar-diffusion-training.md](../pre-training/joint-ar-diffusion-training.md)
**Related:** [_speculative-decoding.md](./_speculative-decoding.md), [dspark.md](./dspark.md)

---

## What it is

Classical speculative decoding needs two artifacts: a small drafter and a large target model. Managing them — training the drafter, keeping its distribution aligned with the target after fine-tuning, serving two model families — is friction. Self-speculation removes the drafter as a *separate* object: one model, two modes.

The prerequisite is that the model was pretrained with a **joint objective** allowing it to operate as both a diffusion (parallel-lookahead) model and an AR (left-to-right) model. Given that, inference dynamically picks the mode based on what the current phase of decoding needs.

## How it works

The joint pretraining objective mixes:

- **AR loss** — standard next-token loss over sequences.
- **Diffusion loss** — mask a block of tokens, train the model to reconstruct all masked positions in parallel, conditioned on the unmasked surrounding context.

At inference, self-speculation runs:

1. **Diffusion draft.** The model in diffusion mode fills a masked block of $k$ tokens in a single forward pass. This is the "draft" — $k$ candidate tokens produced without $k$ sequential passes.
2. **AR verify.** The model in AR mode then scores the $k$-token block sequentially in one *batched* forward pass (positions $1..k$ verified in parallel over the same context). Same standard speculative-decoding accept/resample rule.
3. **Repeat.**

Both modes share weights: the diffusion "draft" and the AR "verify" are the same parameters run under different masking / attention regimes. So the drafter is by-construction distributionally aligned with the verifier — the classical drafter-target KL problem is zero.

## Why it matters

- **One-model deployment.** No separate small drafter to serve, no distribution-alignment drift after fine-tuning the target. Half the operational overhead of two-model speculative setups.
- **Better acceptance than parallel-drafter approaches.** Nemotron-Labs-Diffusion reports beating MTP-K on acceptance rate — because the "drafter" is the same 14B model with the same knowledge, not a small student.
- **Mode-switching for load.** Under high concurrency you can drop to pure AR (small memory per request); under low concurrency, use self-speculation for latency wins. Same weights, no reload.
- **Throughput headroom.** Nemotron-Labs-Diffusion-8B decodes 6× more tokens per forward vs Qwen3-8B (comparable accuracy) → 4× higher throughput on SPEED-Bench with SGLang on GB200.

## Gotchas & tricks

- **Requires joint pretraining from scratch** (or expensive continued pretraining). You can't retrofit self-speculation onto a pure AR checkpoint.
- **Diffusion mode's block size matters.** Larger $k$ → more tokens per draft, but acceptance rate drops if the diffusion model's lookahead calibration is loose. Tuned per model scale.
- **Serving stack needs both mask patterns.** The engine must run either causal (AR) or block-bidirectional (diffusion) attention in the same request. SGLang / vLLM extensions required.
- **AR verify is not free.** You still pay a target forward pass per verify — but you verify $k$ tokens at once.
- **Not the only path.** DSpark achieves similar goals with a *separate* semi-AR drafter; self-speculation trades pretraining cost for serving simplicity.

## Sources

- Paper: *Nemotron-Labs-Diffusion: A Tri-Mode Language Model Unifying Autoregressive, Diffusion, and Self-Speculation Decoding* — Whalen, Garg, Wu, et al., NVIDIA, 2026 — [arXiv:2607.05722](https://arxiv.org/abs/2607.05722).
