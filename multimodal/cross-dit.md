# CrossDiT — Dual-Stream DiT with Bidirectional Cross-Attention
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A paired-modality diffusion transformer architecture that keeps **two parallel DiT streams** — one per modality — and connects them at every block via **bidirectional cross-attention**. Within-stream self-attention preserves each modality's unimodal fidelity; cross-attention carries temporal and semantic alignment between modalities. Introduced in **Kandinsky 6.0 Video** (2026) for joint video+audio generation with lip-sync; the pattern is modality-agnostic.

**Prereqs:** [../fundamentals/attention.md](../fundamentals/attention.md), [../architectures/transformer-block.md](../architectures/transformer-block.md)
**Related:** [../architectures/dynamic-sparse-attention.md](../architectures/dynamic-sparse-attention.md) · [../architectures/multi-head-attention.md](../architectures/multi-head-attention.md) · [../case-studies/kandinsky-6-video.md](../case-studies/kandinsky-6-video.md)

---

## What it is

Joint multi-modality diffusion has typically used one of two patterns: (a) **shared token sequence** — concatenate modality tokens and let a single transformer attend over all of them, or (b) **conditioning adapter** — generate one modality, use features of it to condition a second generator. Pattern (a) spends attention budget uniformly across modalities and tends to destroy unimodal fidelity; pattern (b) is non-symmetric and doesn't model true joint distributions.

**CrossDiT** is the symmetric joint-generation alternative: each modality lives in its own stream (its own DiT backbone with its own self-attention), and the streams communicate only through **bidirectional cross-attention** at each layer. The intra-stream self-attention preserves each modality's internal fidelity; the cross-attention is the sole channel for alignment.

## How it works

- **Two streams.** One DiT backbone per modality, each with its own patch/token embedding and timestep conditioning. In Kandinsky 6.0 Video, modality A is video, modality B is audio; the pattern is agnostic to which pair.
- **CrossDiTFusedBlock.** At each block, after the within-stream self-attention and MLP, insert a **bidirectional cross-attention** step:
  - Queries from stream A, keys/values from stream B → updates to A.
  - Queries from stream B, keys/values from stream A → updates to B.
  Both happen in the same block (hence *bidirectional*), so each stream sees the other's current state.
- **Alignment carried by cross-attention.** Temporal alignment (frame boundary ↔ audio event) and semantic alignment (what's visible ↔ what's audible) are both captured by the same cross-attention weights, which learn to attend to the right spatiotemporal region of the other stream.
- **Timestep sharing.** Both streams share the same diffusion timestep per joint sample, so denoising is synchronized across modalities.

## Why it matters

- **Preserves unimodal fidelity.** Keeping self-attention within-stream means the video stream can keep the inductive biases of its video-only pretraining; same for audio. This is the architectural lever that lets Kandinsky 6.0 Video's **continuous pretraining** work — the video stream, initialized from Kandinsky 5.0, is not destroyed by joint training.
- **Separable warm-up.** Streams can be trained independently (audio from scratch on audio-only; video continued from a pretrained video backbone) and then connected via cross-attention weights initialized from scratch at joint-training time.
- **Reusable template.** Nothing is video/audio-specific. The same pattern applies to any paired-modality diffusion where the modalities need alignment but each has its own fidelity criterion: image+depth, text+speech, video+text, EEG+video, etc.
- **Lip-sync at scale.** Fine temporal alignment (lip motion ↔ phoneme audio) emerges without a dedicated lip-sync module, suggesting the cross-attention paths carry sufficient fine temporal signal.

## Gotchas & tricks

- **Cross-attention weights init from scratch, carefully.** Initialize cross-attention output projections to zero (or small) at the start of joint training so the pretrained within-stream dynamics aren't perturbed until the cross-attention has learned something useful.
- **Memory doubles.** Two streams means ~2× activation memory of a single-stream equivalent at matched parameter count. The compute budget needs to account for this; gradient checkpointing per stream helps.
- **Timestep sharing matters.** If the two streams have different diffusion schedules, the cross-attention has to translate across them, which hurts alignment. Keep schedules aligned by default.
- **Not for strongly asymmetric settings.** If one modality is purely conditioning (e.g., a fixed caption for text-to-image), a plain conditioning adapter is lighter. Use CrossDiT when you're actually *generating* both modalities.
- **Cross-attention sparsity opportunity.** For high-res A/V, cross-attention tokens explode; combines well with content-adaptive sparse attention (see [../architectures/dynamic-sparse-attention.md](../architectures/dynamic-sparse-attention.md)).

## Sources

- Paper: *Kandinsky 6.0 Video: Foundation Models for Synchronized Video and Audio Generation* — Team Kandinsky, 2026 — [arXiv:2610.05608](https://arxiv.org/abs/2610.05608).
- Case study: [kandinsky-6-video.md](../case-studies/kandinsky-6-video.md) — full tech-report breakdown.
