# Frequency-Aware Remasking
*Depth — a test-time scaling method for diffusion language models that commits tokens whose prediction keeps flipping, instead of the most-confident ones.*

**TL;DR:** Diffusion LMs iteratively denoise by committing some tokens and remasking others. The usual rule — commit the tokens the model is most *confident* about — can lock onto a wrong reasoning path when the confident tokens belong to it. Frequency-aware remasking tracks how often each position's prediction *changes* across denoising steps and commits the frequently changing tokens first, so new samples diverge where the model is actually undecided.

**Prereqs:** none (just know what a masked diffusion LM is).
**Related:** [../post-training/reasoning/long-cot-rl.md](../post-training/reasoning/long-cot-rl.md) (hardness-adaptive compute, for AR models).

---

## What it is

A test-time strategy for **diffusion language models** (DLMs) like LLaDA and Dream. Standard DLM decoding picks which tokens to finalize per step using an instantaneous signal — top-k by confidence, bottom-k by entropy. Frequency-aware remasking replaces the signal with a **flip-frequency histogram**: for every position, how often its argmax changed across the denoising trajectory. Positions that keep flipping are the ones the model is uncertain about.

## How it works

Two-stage procedure:

1. **Fast base pass.** Try each question with a cheap certainty-based sampler (e.g. top-k confidence). Accept the ones that solve it.
2. **Frequency-aware pass on the rest.** For the unsolved questions:
   - Track $f_i$ = number of times position $i$'s argmax changed across the T denoising steps.
   - Commit positions in *decreasing* order of $f_i$ — i.e. the ones the model is most undecided about — rather than the most-confident ones.
   - Fresh samples with this remasking schedule diverge at the undecided slots instead of the confident prefix, so sample diversity gets concentrated where it can actually help.

Allocating the extra samples only to the unsolved questions is an adaptive compute knob — easy questions use the base pass's budget, hard questions get the expensive one.

## Why it matters

- **Gives DLMs a real reasoning-time scaling knob.** Autoregressive LMs scale by generating longer CoT; DLMs scale by generating more samples. Frequency-aware remasking makes those samples *actually explore* instead of re-drawing the same confident prefix.
- **Flip frequency is a global signal.** Pointwise confidence sees one step; flip frequency sees the whole denoising trajectory and summarizes which positions the model never committed to.
- **Recovers formerly unsolvable questions.** On LLaDA-8B-Instruct the method solves 3.75–18.75% of questions the certainty baseline couldn't solve in 100 samples — a non-trivial regime extension.

## Gotchas & tricks

- **Requires access to intermediate predictions,** not just the final decoded sequence. Any wrapper that exposes only the output must be instrumented.
- **The flip count's magnitude depends on T** (number of denoising steps). Normalize before comparing across configurations.
- **On easy questions, frequency-aware decoding can underperform** certainty-based decoding — which is why the method keeps the two-stage structure. Don't replace the base pass; add the frequency pass as a fallback.
- **Composes with any adaptive-allocation scheme.** The paper layers it on top of an adaptive sampler that chooses how many samples each question gets; both signals are useful.

## Sources

- Paper: *Test-Time Scaling for Diffusion Language Models with Frequency-Aware Remasking* — Yue Yu, Dongruo Zhou, Yinglun Zhu, UC Riverside / Indiana University, 2026 — [arXiv:2611.10486](https://arxiv.org/abs/2611.10486).
