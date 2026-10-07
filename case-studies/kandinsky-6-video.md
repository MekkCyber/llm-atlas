# Case Study: Kandinsky 6.0 Video

*An open-weights family of foundation diffusion models (3B Lite + 29B Pro) that generates 5-second video clips with synchronized 44 kHz audio — including lip-sync — in both text-to-A/V (T2AV) and image-to-A/V (I2AV) modes, with a built-in super-resolution model raising output to full-HD. The interesting part is the **CrossDiT dual-stream architecture** and the **continuous audio-video pretraining recipe** that makes joint A/V foundation modeling tractable without sacrificing unimodal fidelity.*

**Related concepts:** [cross-dit](../multimodal/cross-dit.md) · [dynamic-sparse-attention](../architectures/dynamic-sparse-attention.md) · [mla](../architectures/mla.md) · [attention](../fundamentals/attention.md) · [grpo](../post-training/grpo.md) · [rejection-sampling](../post-training/rejection-sampling.md)

---

## What this is

Kandinsky 6.0 Video, released October 2026 by Team Kandinsky. Two foundation models in the same family:

- **Kandinsky 6.0 Video Lite** — 3B parameters, same architecture, faster.
- **Kandinsky 6.0 Video Pro** — 29B parameters, best quality.

Both generate **5-second 44 kHz synchronized audio-video clips** including lip-sync, in two modes:

- **T2AV** — text prompt → audio + video.
- **I2AV** — image prompt → audio + video (image provides visual grounding).

A separate built-in **super-resolution** module raises output to **1920×1080** (full-HD). Code, weights, and HuggingFace Diffusers integration are released under MIT.

The paper matters because it bundles: (1) the **CrossDiT** dual-stream architecture, (2) a **continuous audio-video pretraining strategy** that trains the audio stream first and then jointly without destroying video quality, and (3) a complete production pipeline (pretrain → SFT → RL → distillation) applied end-to-end at 29B scale and released openly. Any one of these would make the paper. Together they establish an open reference for A/V foundation modeling.

---

## Architecture at a glance

```
Dual-stream CrossDiT:

  Video stream  ──┐
   (pretrained from   │   bidirectional cross-attention
    Kandinsky 5.0)    │   at each CrossDiT block
                      ├─────────── CrossDiTFusedBlock ───────────
  Audio stream  ──┘        - cross-attention aligns temporal + semantic
   (newly trained)         - within-stream self-attention preserves fidelity

Downstream:
  → T2AV / I2AV diffusion head
  → separate SR latent-upscaler for 1080p output
```

The video stream is initialized from the pretrained Kandinsky 5.0 Video backbone; the audio stream is trained from scratch. The two streams communicate at every CrossDiT block via **bidirectional cross-attention**, which carries temporal alignment (frame boundaries ↔ audio events) and semantic alignment (what's visible ↔ what's audible). See [cross-dit](../multimodal/cross-dit.md).

---

## Training recipe

### Stage 1 — Audio-stream warm-up

Train the audio stream alone on large-scale audio corpora. The video stream is kept frozen at its pretrained weights. This gives the audio stream a competent prior over 44 kHz audio distributions before any joint training touches it.

### Stage 2 — Joint audio-video continuous pretraining

Train both streams jointly on paired audio-video data. Critically, this stage is **continuous pretraining**, not re-initialization: the video stream was already a strong pretrained model and must not regress. The cross-attention weights are added at this stage and trained from scratch; both streams' self-attention weights continue training. The paper reports that joint training preserves unimodal fidelity on video-only prompts while learning synchronized A/V generation.

### Stage 3 — Supervised fine-tuning (SFT)

Curated prompt–A/V pairs for both T2AV and I2AV modes. The image-conditioning path for I2AV is introduced at SFT.

### Stage 4 — Reinforcement-learning post-training

Preference-based RL (per the paper) applied to the full A/V generation, with human-side-by-side preferences as the signal — standard for RLHF on generative models, applied here to the joint output rather than audio or video alone.

### Stage 5 — Distillation

Few-step distillation for inference acceleration. The super-resolution module is applied as a separate latent-upscaler at the end.

---

## Super-resolution

A dedicated **SR DiT** latent-upscaler takes the generated video latents and produces the full-HD (1920×1080) output. The super-res stage is trained independently of the main CrossDiT. This decouples the quality of the generative stage from the final-resolution target, letting the 29B model generate at its training resolution and then upscale.

---

## Evaluation snapshot

Side-by-side human evaluation vs. (a) Kandinsky 5.0 Video Pro (predecessor) and (b) leading audio-video generation systems:

- **vs. Kandinsky 5.0 Video Pro**: clear preference for 6.0 Pro.
- **vs. leading A/V generators**: competitive overall; the paper highlights **speech-quality parity** as a particular strength (where lip-sync and A/V alignment are harder for most baselines).

Quantitative benchmark numbers follow the paper's own A/V quality + alignment metrics; the headline is that an open 29B model is competitive with the state of the art and ships weights.

---

## Key takeaways

1. **CrossDiT is a reusable recipe for paired-modality diffusion.** Dual stream + bidirectional cross-attention is more general than video+audio; the same template applies to image+depth, text+speech, video+text, or any pair where modalities need temporal/semantic alignment. See [cross-dit](../multimodal/cross-dit.md).

2. **Continuous pretraining preserves unimodal fidelity.** The two-stage approach (audio warm-up first, then joint) is what prevents the video stream from degrading on video-only prompts while learning joint generation. The lesson generalizes to any multimodal continual-training scenario.

3. **Full pipeline is open.** Not just weights — code, Diffusers integration, and the full pretrain → SFT → RL → distillation recipe. Compare to closed A/V stacks (Sora, Veo) where the training recipe is opaque.

4. **Lip-sync at 29B is a quality milestone.** Lip-sync has been the hardest alignment problem in A/V generation; getting it right at open-source scale is a signal that CrossDiT's cross-attention paths carry enough fine temporal information.

5. **Super-resolution decoupling pays off.** Training the generative DiT at lower res + a dedicated SR module uses compute more effectively than training the generative model directly at 1080p.

6. **5-second clips are the current sweet spot.** The window is long enough for meaningful motion + speech, short enough to keep the attention budget tractable. Longer-horizon A/V generation is the natural next frontier (see [in-distribution forcing](../inference/id-forcing.md) for the long-video stability problem).

---

## What's still opaque

- **Exact hyperparameters** for RL stage (reward model, preference pair construction) are summarized but not fully disclosed.
- **Training-data mix**: described in categories but the full corpus is not published.
- **SR-model architecture detail** is referenced in figures but not fully specified in the abstract.
- **Lip-sync evaluation** is reported but no public lip-sync-specific benchmark number is given — comparison relies on human eval.

---

*Pairs well with:* [cross-dit](../multimodal/cross-dit.md) for the architecture depth; [dynamic-sparse-attention](../architectures/dynamic-sparse-attention.md) for a complementary efficiency trick at the attention level for joint A/V training.

## Sources

- Paper: *Kandinsky 6.0 Video: Foundation Models for Synchronized Video and Audio Generation* — Team Kandinsky, 2026 — [arXiv:2610.05608](https://arxiv.org/abs/2610.05608).
- Predecessor: Kandinsky 5.0 Video — Team Kandinsky, 2025.
- Code & weights: released by Team Kandinsky under MIT (per paper).
