# Speculative Forcing
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Speculative decoding, applied to the *denoising-step* axis of few-step autoregressive video generators. Each generated token lives on an $N$-step denoising schedule; speculative forcing treats intermediate denoising outputs as drafts, freezes tokens once they stop changing, and uses two verifiers (a training-free similarity-and-stability check and a learned per-token residual predictor) to decide when to stop refining a given token. Different tokens exit at different denoising depths.

**Prereqs:** [speculative-decoding.md](./speculative-decoding.md)
**Related:** [../fundamentals/attention.md](../fundamentals/attention.md)

---

## What it is

Autoregressive video generators like LongLive and Lingbot-World produce each video token with a few denoising steps (e.g., 4-step diffusion heads). Most tokens actually converge long before step 4 — the extra steps are wasted. Speculative Forcing accepts this empirically and adds a *per-token* early-exit test over the denoising axis.

The analogy to speculative decoding is strict: the "draft" is an under-denoised token (a token with few remaining noise iterations); the "verifier" accepts the draft once its predicted remaining change is small.

---

## How it works

For an $N$-step denoising schedule and a token with trajectory $\tilde{z}_0, \tilde{z}_1, \ldots, \tilde{z}_N$ (noisier → cleaner):

### Training-free verifier (similarity + stability)

At each denoising step $k$, decide to freeze the token if:

- **Similarity to clean content.** The current prediction $\tilde{z}_k$ is sufficiently close to a reference clean estimate.
- **Stability across steps.** The change between $\tilde{z}_{k-1}$ and $\tilde{z}_k$ is below threshold.

Both metrics are cheap to compute from the generator's own activations.

### Learned verifier (residual predictor)

A small per-token head predicts the **remaining change** $\|\tilde{z}_N - \tilde{z}_k\|$ from the generator's hidden features at step $k$. If the predicted residual is below threshold, freeze. Trained once as a drop-in head on sample trajectories.

### Scheduling

Each token has its own exit step. The generator's forward pass over tokens treats frozen tokens as fixed context (no further compute) and continues denoising only the unfrozen ones.

---

## Why it matters

- **Speedup without quality drop.** On LongLive: matches 4-step quality at ~1.7 denoising steps per token (>1.5× speedup). On Lingbot-World 2.0: ~2.5 steps per token for 1.35× speedup.
- **Transfers the speculative-decoding mental model** into diffusion-flavored generation. Shows the draft-verify pattern isn't attention-specific — any iterative-refinement generation process has slack to exploit.
- **Orthogonal to attention or model compression.** The savings are on the step axis; cache-side and head-side optimizations compose freely.

---

## Gotchas & tricks

- **Thresholds trade quality for speed.** Aggressive early-exit thresholds give larger speedups but can show up as subtle artifacts (banding, residual noise). The paper tunes on a validation set.
- **Learned verifier beats training-free in the small-step regime.** When $N$ is small (4 steps), one wrong-exit decision is expensive; the learned residual predictor is more precise there.
- **Not a replacement for distillation.** Few-step diffusion already compresses the denoising schedule (via distillation / consistency objectives). Speculative forcing is the *per-token* complement, applied on top.
- **Needs per-token features.** Models that pool over tokens before denoising (some video DiTs) require a small refactor to expose per-token hidden states for the verifier.

---

## Sources

- Paper: *Speculative Forcing for Efficient Autoregressive Video Generation* — anonymous HF submission (one contributor at NAVER AI Lab), 2026 — arXiv [2610.22347](https://arxiv.org/abs/2610.22347).
- Baseline: *LongLive* — few-step AR video generator.
- Baseline: *Lingbot-World 2.0* — AR video world-model.
- Precursor: *Fast Inference from Transformers via Speculative Decoding* — Leviathan et al., 2022/23 — see [speculative-decoding.md](./speculative-decoding.md).
