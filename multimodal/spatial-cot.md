# Spatial chain-of-thought for VLMs (SpatialSpeak)
*Depth — a two-stage pipeline: QA-native multi-view reconstruction pretraining followed by spatial CoT supervision.*

**TL;DR:** VLMs trained on final answers alone get no direct supervision on the geometric intermediates their spatial reasoning depends on. SpatialSpeak fixes this by first pretraining the VLM to *reconstruct* local geometry and global scene layout from multi-view images in a QA-native format, then adding spatial CoT supervision on top so the reasoning head can consume those geometric features quantitatively.

**Prereqs:** [../post-training/reasoning/long-cot-rl.md](../post-training/reasoning/long-cot-rl.md), [README](README.md)
**Related:** [interleaved-reflection-rl](interleaved-reflection-rl.md)

---

## What it is

A training recipe for vision-language models focused on spatial reasoning tasks — multi-view scene understanding, quantitative distance/direction questions, and cross-view referring. Its contribution is showing that the answer-only training used by most VLMs undertrains the *intermediate* geometric estimates that spatial CoT relies on, and giving a fix.

## How it works

- **Stage 1: QA-native reconstruction pretraining.** Given multi-view images of a scene, the VLM is trained to answer reconstruction questions in the *same QA format* the downstream task will use — questions like "what is the depth at pixel `(u, v)` in view 2?", "where is the origin of view 1 relative to view 2?", "what class of object occupies the region at `(x, y, z)`?". Outputs live in the QA vocabulary, so the model doesn't need a separate reconstruction head — the reconstruction *is* text emission.
- **Stage 2: Spatial CoT training.** Now the model is trained with chain-of-thought supervision on quantitative spatial tasks. The CoT tokens can directly reference the geometric predicates the model learned to emit in stage 1 (depths, poses, layouts), giving a grounded intermediate representation.
- Both stages use the same backbone and same vocabulary; the pretraining objective is a QA task the CoT stage can *quote*.

## Why it matters

Spatial reasoning has been the stubborn weak spot of VLMs even as text CoT and code CoT saturated their respective benchmarks. Making the geometric intermediates first-class training targets — not just implicit hidden state — is a general pattern: any domain where CoT tokens should refer to structured latent features (chemistry, tables, diagrams) can borrow the same "QA-native intermediate pretraining" trick. Reported result: SOTA on ReVSI, VSI-Bench, and SPAR-Bench with ~+8.7-point gains over answer-only baselines; ablations confirm the reconstruction pretraining is what carries the gain.

## Gotchas & tricks

- **QA format is load-bearing.** If reconstruction is trained with a separate head or a different vocabulary, the CoT stage cannot consume its outputs and gains collapse to noise.
- **Multi-view coverage.** The recipe assumes multi-view input; a single image collapses reconstruction supervision to depth prediction, which is a much weaker signal.
- **Rendering-time cost.** CoT that quotes explicit geometric predicates is longer than answer-only inference. Budget it — spatial CoT typically doubles output length.
- Interacts with [interleaved-reflection-rl](interleaved-reflection-rl.md): a unified model can, in principle, *render* an intermediate diagram of its geometric estimate to further ground the CoT.

## Sources

- Paper: SpatialSpeak: QA-Native Reconstruction with Local and Global Context for Spatial Chain-of-Thought Reasoning — Cao, Zhang, Chen, Zhong, Gao, Hong, Xu, 2026 — [arXiv:2609.33616](https://arxiv.org/abs/2609.33616)
