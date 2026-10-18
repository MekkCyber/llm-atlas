# Speculative Forcing
*Depth — cross-step speculative decoding for few-step autoregressive video.*

**TL;DR:** Few-step autoregressive video generators run every token through the full denoising schedule, even when many tokens are already close to their final value by step 2 or 3. Speculative Forcing treats intermediate clean estimates as *draft tokens*, accepts converged ones early, and freezes them — while keeping their KV cache live as context for later tokens. Two verifiers decide acceptance: a training-free similarity+stability check, or a learned verifier that reads intermediate features.

**Prereqs:** [README.md](README.md)
**Related:** [attention.md](../fundamentals/attention.md)

---

## What it is

Speculative decoding, but *across denoising steps* instead of across models. The "draft" is yourself at an earlier step; verification is "have you already converged." Unlike classic model-pair speculative decoding, there's no second small model — the draft and verify are both the generator at different step indices.

## How it works

For a few-step AR generator with steps `t = T, T−1, …, 0`:

1. At intermediate step `t`, read the current clean estimate per token.
2. **Training-free verifier.** For each token, check (a) similarity to the previous step's clean estimate (stability) and (b) similarity to the eventual projected clean content. If both pass, accept the token at step `t`.
3. **Learned verifier.** Alternatively, a small head reads the generator's intermediate features and predicts "accept now."
4. Accepted tokens are frozen for subsequent steps. Their KV caches remain available as context so the still-denoising tokens still condition on them.

## Why it matters

Few-step AR video generators are the current efficient-video frontier (LongLive, world models). Speculative Forcing reports ~1.7 steps/token (vs. nominal 4) on LongLive, 1.5× end-to-end speedup, no quality loss; 2.5 steps/token on Lingbot-World 2.0 at 1.35×. The idea — early-exit per token with a cheap verifier — likely transfers to any iterative-refinement generator.

## Gotchas & tricks

- Freezing a token early commits to its value; wrong early-accepts propagate. Verifier thresholds control the false-accept rate — tune on validation.
- Accepted tokens' KV caches stay live, so memory doesn't actually shrink; the gain is compute, not VRAM.
- Training-free verifier is cheaper but less accurate; the learned one adds a small head and a training pass. Pick based on serving constraints.

## Sources

- Paper: *Speculative Forcing for Efficient Autoregressive Video Generation* — authors not fully disclosed on HF page, NAVER AI Lab, 2026 — [arXiv:2610.22347](https://arxiv.org/abs/2610.22347)
