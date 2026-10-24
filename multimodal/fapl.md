# Factor-Aware Predictive Learning (FAPL)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A three-stage training recipe that forces a Video Diffusion Transformer (DiT) to actually use an MLLM's latents as its conditioning signal, then teaches factor-aware control (character, scene, action, camera) and temporal inference from sparse inputs on top. Without this, the DiT bypasses MLLM latents through its native conditioning channels and fails at compositional cinematic generation (HKUST / Tencent Video, 2026).

**Prereqs:** [../fundamentals/attention.md](../fundamentals/attention.md)
**Related:** [README.md](README.md)

---

## What it is

A training framework for cinematic video generation where (a) the DiT must be *forced* to use MLLM latents (otherwise it leans on text/image conditioning and ignores the richer MLLM signal), and (b) factor-aware control over character / scene / action / camera must be learned without destroying temporal coherence. FAPL achieves both via three complementary learning signals.

## How it works

Stage 1 — **Latent-only generator grounding.** Freeze the MLLM and connector. Train the DiT to generate video using *only* MLLM latents as conditioning (strip native text/image inputs). This bolts the DiT to the MLLM's latent space before any factor training begins.

Stage 2 — **Factor-span replacement.** Pair training videos such that one factor (say, camera) changes between the pair while the others are preserved. Train the DiT to realise exactly that target-only change given the paired MLLM latents. The "factor span" is the subset of latent tokens that encode the factor being changed; the loss encourages precisely those tokens to drive the diff.

Stage 3 — **Temporal predictive training.** Freeze the DiT. The MLLM + connector see two kinds of observations sharing a video target: (i) text only, (ii) text plus the complete video. Train them through the frozen DiT so that limited-input latent trajectories still generate the target, while retaining complete-observation reconstruction as a regulariser.

## Why it matters

- **Breaks the bypass pathology.** DiTs with multiple conditioning inputs tend to route through whichever is easiest and ignore MLLM latents. Latent-only grounding is a cheap way to force investment in the richer channel.
- **Compositional cinematic control.** Factor-span replacement is a clean mechanism for "change camera, keep character and scene"-style edits, which prior video models typically couldn't disentangle.
- **Sparse → video.** Stage 3 gives sparse-input generation (text only, or 1–2 keyframes) by training the MLLM to produce latent trajectories the frozen DiT can realise.

## Gotchas & tricks

- **Pair data is the bottleneck.** Factor-span replacement needs videos that differ on exactly one factor. These are rare in the wild; the paper leans on synthetic / curated pairs.
- **Stage ordering is strict.** Running factor control before latent grounding re-opens the bypass — the DiT just learns the factor from text, not from latents.
- **Factor spans must be identified in latent space.** The paper assigns per-token factor roles; this is an architectural choice that other MLLMs would need to replicate.
- **Reconstruction regulariser matters.** Dropping the complete-observation branch in Stage 3 drifts the MLLM representations away from the DiT's expected input distribution.

## Sources

- Paper: *Making MLLM Latents Matter: Factor-Aware Predictive Learning for Cinematic Video Generation* — Wang et al., HKUST / Tencent Video / École Polytechnique, 2026 — [arXiv:2610.30926](https://arxiv.org/abs/2610.30926).
