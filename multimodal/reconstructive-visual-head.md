# Reconstructive Visual Head

*Depth — auxiliary MLLM head that predicts the *latent* representation of the transformed visual state, giving temporal-reasoning training a target that isn't a natural-language sentence.*

**TL;DR:** MLLMs on video-reasoning tasks tend to paraphrase captions rather than track state across frames, partly because the training objective is always a natural-language answer. The Reconstructive Visual Head (introduced by ChronoVision, 2026) is an auxiliary output head that predicts the *latent representation* of the final transformed visual state alongside the natural-language answer. Trained end-to-end with SFT, it shapes the model's internal representations toward "what should the visual state look like now," and pairs cleanly with an RL stage that rewards latent-space alignment as an implicit process signal.

**Prereqs:** *(basic MLLM familiarity)*
**Related:** [README.md](README.md), [../post-training/rlvr.md](../post-training/rlvr.md), [../post-training/reasoning/prm.md](../post-training/reasoning/prm.md)

---

## What it is

An **extra output head** attached to an MLLM that emits a *visual latent* target parallel to the ordinary text-decoding head. Two components in the ChronoVision recipe use it:

- **Reconstructive Visual Head.** During SFT, predicts the latent of the final visual state after the described transformation. Loss = distance in latent space (cosine or MSE) to the ground-truth latent.
- **ROI Attention Locating.** A module that focuses attention on task-relevant regions, keyed by semantic span queries (e.g., "which region is being manipulated"). Provides the spatial priors the reconstructive head needs.

At post-training, RL adds an implicit process-grounding term: reward includes a latent-alignment component alongside outcome correctness.

---

## How it works

1. **SFT stage.**
   - Forward: image + instruction → hidden states → (text head, reconstructive visual head, ROI attention).
   - Text head loss: standard AR cross-entropy on the natural-language answer.
   - Visual head loss: distance from predicted latent to ground-truth latent of the transformed image.
   - ROI attention loss: attention weights matched to region annotations (semantic span queries).
2. **RL post-training stage.**
   - Composite reward: outcome correctness + latent-process alignment + unsupervised visual-focus term.
   - The latent-alignment term acts as an implicit PRM — it grades whether intermediate reasoning corresponds to the actual visual transformation, without needing per-step human labels.
3. **Inference.** The reconstructive head is not needed at inference; it's a training-time scaffold. The trained model uses only its text head and (implicitly) the reshaped internal representations.

## Why it matters

- **Bypasses the "reason in words" bottleneck.** Language-based reasoning is a lossy encoding of continuous visual change; predicting the latent directly is a more faithful supervision target.
- **A cheap alternative to PRMs for visual reasoning.** Latent alignment is the process signal — no per-step human labels required, no separate PRM to train.
- **Concrete transfer.** ChronoVision reports 74.8% in-domain and 71.6% out-of-domain on Vbvr-VQA (a video-reasoning benchmark cast as image ordering), plus 55.0% on IntPhys2 — both SOTA at report time.

## Gotchas & tricks

- **Latent space matters.** The head predicts *some* latent (a VAE / autoencoder embedding of the target frame); choice of encoder determines what "close in latent space" means. Reconstruction losses in a bad latent space penalize the wrong things.
- **Training-only.** The head is a scaffold; production inference doesn't run it. Deployment cost is unchanged.
- **Needs ground-truth transformed states.** Synthetic or renderer-based data trivially provides these; naturalistic video needs paired before/after frames, which limits data.
- **ROI attention conflict with existing attention regularizers.** Stack cautiously; competing regularizers can leave the model attending to nothing.
- **Latent alignment can be reward-hacked.** The RL policy can find outputs that produce "close-enough" latents without doing the reasoning; keep a KL to a reference policy and audit with held-out prompts.

## Sources

- Paper: *ChronoVision: Temporal Reasoning via Latent State Reconstruction* — Shen et al., 2026 — [arXiv:2608.05631](https://arxiv.org/abs/2608.05631). Introduces Vbvr-VQA and reports SOTA on Vbvr-VQA and IntPhys2.
