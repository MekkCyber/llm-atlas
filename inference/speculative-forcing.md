# Speculative Forcing
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Few-step autoregressive video generators (LongLive, Lingbot-World) run every token through every denoising step, even when some tokens have already converged. Speculative Forcing treats each step's **intermediate clean estimate** as a *draft* and uses a verifier to accept it early, freezing that token (keeping its K/V in context) while the rest keep refining. The speculative-decoding idea, moved from language models to diffusion.

**Prereqs:** [../fundamentals/attention.md](../fundamentals/attention.md)
**Related:** [../pre-training/mtp.md](../pre-training/mtp.md)

---

## What it is

A video generator typically denoises $N$ steps for every token. Speculative Forcing observes that in a *few-step* generator (where $N$ is already small, 2–4), different tokens converge at different rates: smooth regions and background stabilize quickly; motion / occlusion boundaries need the full budget. Uniformly denoising everything to step $N$ is wasted compute on the easy half.

The method keeps the base generator unchanged and inserts an **early-exit verifier** between denoising steps. A token's intermediate clean estimate is accepted as its final value when the verifier says so; the rest continue.

## How it works

At each denoising step $k$ for each token $i$:

1. Produce the step-$k$ clean estimate $\hat x_i^{(k)}$ (the generator already computes this as part of noise prediction).
2. **Verifier decides** whether to freeze $x_i := \hat x_i^{(k)}$ or keep refining:
   - **Training-free verifier.** Accept if $\|\hat x_i^{(k)} - \hat x_i^{(k-1)}\|$ (step-to-step stability) is below a threshold *and* the estimate's similarity to prior clean content is high.
   - **Learned verifier.** A small head reads the frozen generator's intermediate features and predicts the remaining gap to the full-step output; accept when predicted gap is below threshold.
3. Frozen tokens keep their keys/values in the attention cache so later refinement for other tokens still sees them as context — the attention trajectory for the unfrozen tokens is identical to what it would have been.

The generator's weights are not touched; the training-free verifier is purely inference-time, the learned one is a small side head.

## Why it matters

- **Composes with step reduction.** Consistency / DMD / LCM style step-reduction already shrinks $N$; Speculative Forcing shrinks the *per-token* work inside those $N$ steps. Multiplicative.
- **Quality preserved by construction.** Frozen tokens were produced by the generator itself — the verifier only decides *when* each becomes final. Short of verifier mistakes, the output distribution is unchanged.
- **Fits streaming generation.** Freezing tokens means they can be flushed downstream earlier — a real latency win for interactive video, not just a FLOPs win.

## Gotchas & tricks

- **Verifier false accepts are the only risk.** A premature freeze on a motion edge visibly degrades the frame. The training-free verifier is more conservative (lower throughput, safer); the learned verifier gets higher throughput but needs calibration per generator.
- **Keep K/V for frozen tokens.** The whole point of freezing (vs resampling) is that cached attention context stays available to later tokens; a naive implementation that drops frozen tokens from the cache corrupts spatial coherence.
- **Measured on few-step bases.** Reported on LongLive and Lingbot-World 2.0 — bases that already use 2–4 steps. Behavior on long-schedule generators (50+ steps) isn't demonstrated and the uniform-work assumption may hold less strongly there.
- **Latency ≠ throughput on video.** The paper reports fps gains (41.7→1.5× etc.); for interactive applications the latency-to-first-token matters more and depends on which tokens freeze first.

## Numbers

| Base | Steps/token w/ SF | Speedup | FPS |
| --- | --- | --- | --- |
| LongLive (4-step) | ~1.7 | 1.5× | 41.7 / 37.8 |
| Lingbot-World 2.0 | ~2.5 | 1.35× | 24.5 |

## Sources

- Paper: *Speculative Forcing for Efficient Autoregressive Video Generation* — NAVER AI Lab et al., 2026 — [arXiv:2610.22347](https://arxiv.org/abs/2610.22347).
- Code: https://github.com/naver-ai/speculative-forcing
- Inspiration: speculative decoding (Leviathan et al., 2023); Medusa (Cai et al., 2024) — see [../pre-training/mtp.md](../pre-training/mtp.md) for the LM-side analogue.
