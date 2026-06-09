# dots.tts — continuous autoregressive TTS with self-corrective post-training
*Depth — a 2B-parameter open TTS foundation model that pairs continuous AR latents, full-history flow-matching, and reward-free self-correction.*

**TL;DR:** dots.tts is an Apache-2.0 2B-param TTS model in the continuous-autoregressive family. Three deltas vs. prior continuous-AR-TTS: an AudioVAE trained with multiple objectives for a semantically structured speech latent, full-history conditioning in the flow-matching head, and a reward-free self-corrective post-training pass on the head. SOTA among open TTS on Seed-TTS-Eval, with 85 ms first-packet latency after MeanFlow distillation.

**Prereqs:** [attention](../fundamentals/attention.md), [rejection-sampling](../post-training/rejection-sampling.md)
**Related:** [meanflow-distillation](../inference/meanflow-distillation.md)

---

## What it is

A foundation TTS model that generates speech autoregressively over a continuous (not discretised) latent space. AR over continuous tokens is a less-explored design point than discrete-codec AR; dots.tts argues the gains are real, given the right latent and post-training.

## How it works

Three pieces:

1. **AudioVAE.** Encodes raw audio into a continuous latent space, trained with multiple objectives — reconstruction + semantic alignment + a prediction-friendly regulariser — so that the latent is both perceptually faithful and easy for an autoregressive head to predict.
2. **Flow-matching head with full-history conditioning.** The decoder at step `t` conditions on *all prior* latents (not just the previous one), preserving long-range consistency and reducing drift across long utterances.
3. **Reward-free self-corrective post-training.** Show the model its own broken samples and reward it for fixing them — no preference labels, no learned reward. A self-distillation-flavoured loop that improves robustness and acoustic quality without external supervision.

After post-training, *CFG-aware MeanFlow distillation* compresses the flow-matching head into a low-step inference graph, dropping first-packet latency to 85 ms (output streaming) / 54 ms (dual streaming).

## Why it matters

- Open-weights audio LMs are a sparse market; an Apache-2.0 2B with deployable latency closes that gap.
- The reward-free self-corrective post-training trick is recipe-portable: any generative head with a fixable distribution of failures could try it.
- Demonstrates that continuous-AR + flow-matching is competitive with discrete-codec TTS *if* the latent and post-training are right.

## Gotchas & tricks

- **AudioVAE objectives must trade off.** Pure reconstruction yields a "lossy waveform" latent that's hard to predict; pure semantic alignment loses prosody. The published recipe weights both.
- **Full-history conditioning is expensive.** Mitigated by the MeanFlow distillation pass for inference, but training cost is non-trivial.
- **Self-corrective training can collapse modes.** Mixing in clean data prevents the model from learning to "fix" all samples into one style.

## Sources

- Paper: *dots.tts Technical Report* — Lian, Li, Li, Wang, Zheng, Tian, Ma, Zhang, Yu — 2026 — [arXiv:2606.07080](https://arxiv.org/abs/2606.07080)
