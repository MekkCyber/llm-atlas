# Native Joint Audio-Video Generation
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Diffusion-based generation of *synchronized* audio+video from a single joint model rather than a video model followed by a separately-trained audio model. Two-half backbone: streams processed independently in the first half, coupled through **Gated Cross-Modal Attention** (per-head, per-token output gates) in the second. Reinforcement-learning stage routes video-, audio-, and cross-modal rewards to the corresponding streams (**Modality-Aware Multimodal Feedback**). A separate 1-step autoregressive 2K refinement, distilled from a bidirectional multi-step teacher, delivers real-time 2K output.

**Prereqs:** [../fundamentals/attention.md](../fundamentals/attention.md)
**Related:** [../post-training/_rl.md](../post-training/_rl.md)

---

## What it is

Two-stage pipelines (video first, then audio conditioned on video) can't model *reciprocal* dynamics — audio events don't inform video generation. Native joint models produce both streams in one diffusion process, letting each modality condition on the other's evolving prediction. The challenge is preventing early collapse where one modality's easier reconstruction dominates the shared representation.

## How it works

**Backbone: half-independent, half-coupled.**
- **First half:** two modality-specialized towers denoise audio and video streams in parallel with *no* cross-attention. Each learns clean, modality-native features.
- **Second half:** Gated Cross-Modal Attention layers couple the streams. Each cross-attention head has an **output gate** (per-token, per-head) that decides how much of the cross-modal signal is written into the target modality's residual stream. Gates are learned end-to-end.

$$
\text{out}_i = g_i \cdot \text{CrossAttn}(x^{(A)}, x^{(V)}) + (1 - g_i) \cdot 0
$$

so the network can *skip* cross-modal signal at positions where the modalities are locally uncorrelated (silent frames, static shots).

**Conditioning.** First frame + text prompt. The generator jointly denoises 5–10s clips.

**Data pipeline.** A Unified Audio-Video Data System constructs temporally-coherent clips, generates structured multimodal annotations, and organizes clips into capability-oriented pools (action, dialogue, ambient, etc.).

**Training stages.**
1. Two audio-video pre-training stages.
2. High-Quality Finetuning on a curated top-tier subset.
3. **Audio-Video Reinforcement Learning** with **Modality-Aware Multimodal Feedback**: video reward → video branch, audio reward → audio branch, cross-modal (sync/alignment) reward → coupling layers. Prevents the "loudest reward wins" pathology in flat multimodal RL.

**Autoregressive 1-Step 2K Refinement.** Separate refinement model: a bidirectional multi-step 2K teacher is adapted into an autoregressive multi-step refiner, then distilled into a student that performs *one* denoising evaluation per temporal chunk. Real-time 2K output at inference.

## Why it matters

- **Reciprocal modality dynamics.** Native joint modeling lets an off-screen sound cue drive a corresponding visual event and vice versa — impossible in cascaded pipelines.
- **Gated cross-attention is a broadly reusable pattern.** Any modality-fusion architecture where fusion strength should vary by position benefits from the same output-gate trick — not confined to audio-video.
- **Modality-aware RL routing** solves a general multimodal-RLHF problem: gradient credit assignment across streams with different reward scales.
- **Public 2K real-time baseline.** DreamX-Creator 1.0 delivers 2K joint A/V generation at compute budgets that make consumer/edge deployment plausible.

## Gotchas & tricks

- **Gate initialization matters.** Initialize gates near zero so the model starts near modality-independent and *learns* to couple. Uniform-high gates cause the early collapse this design was meant to prevent.
- **Reward-routing bugs are silent.** If your video reward accidentally backprops into audio-branch params, you'll see subtle audio-quality regressions with no obvious cause. Assert gradient paths in unit tests.
- **1-step distillation trades quality for latency.** The refinement student's outputs are visibly slightly softer than the multi-step teacher's — the audio-visual sync is preserved, texture detail is not.
- **Sync-reward design is delicate.** Cross-modal reward that only measures gross correlation misses fine-grained sync (lip-sync, foley timing). Use a specialized sync verifier if that matters.
- **Data-side sync errors compound.** A/V clips with even 40–60 ms alignment error in training data teach the model to be similarly loose. Filter aggressively.

## Sources

- Paper: *DreamX-Creator: Democratizing Native Audio-Video Generation at 2K Resolution* — Zhu et al. — Meituan, 2026 — arxiv.org/abs/2608.31106.
