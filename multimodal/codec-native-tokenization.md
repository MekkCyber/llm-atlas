# Codec-Native Visual Tokenization
*Depth — reuse video codec motion vectors and residuals to steer VLM token density.*

**TL;DR:** Video VLMs typically tokenize sparse keyframes uniformly. That's fine for slow reasoning over pre-recorded clips, disastrous for real-time streams where token throughput is the bottleneck. **Codec-native tokenization** piggybacks on the *video encoder's own* motion vectors and residual energy — signals the codec has already computed — to allocate more tokens to dynamic regions and fewer to static ones. The result is a smaller token stream at matched temporal fidelity.

**Prereqs:** [README](README.md)
**Related:** [video-world-models](video-world-models.md)

---

## What it is

A visual tokenizer for VLMs whose per-region density is driven by codec-side signals rather than a uniform grid or a keyframe-only strategy. Concretely:

- **Motion vectors** (from H.264/H.265-style block motion estimation) indicate which regions *changed* between frames.
- **Residual energy** (the reconstruction error after motion compensation) indicates which regions *changed in a way motion alone doesn't explain* — the visually informative bits.

Regions with high motion or residual get tokenized densely; regions that are static or well-predicted by prior frames get few or no new tokens.

## How it works

```
for each incoming frame f_t:
    mv, resid = codec.encode_frame(f_t | f_{t-1})       # standard codec output
    saliency = motion_score(mv) + residual_score(resid) # per spatial block
    density_map = quantize(saliency, levels)
    tokens = adaptive_vit(f_t, density_map)             # tokenize per density
    vlm.consume(tokens)
```

The VLM sees a heterogeneous token stream: dense tokens where the scene is moving or changing unpredictably, sparse tokens elsewhere. Temporal context is preserved because the model sees *why* the codec allocated bits, not just downsampled pixels.

## Why it matters

- Streaming video reasoning is throughput-bound on tokens per second, not model FLOPs. Cutting tokens is the direct lever.
- Codec signals are essentially free — they're computed anyway to compress the video.
- Aligns token density with what a human vision system also allocates attention to (motion and unpredicted change).

## Gotchas & tricks

- **Codec parameters leak into token statistics.** H.264 at low bitrate produces different motion vectors than at high bitrate for the same content. Retrain the tokenizer per codec / bitrate regime or normalize.
- **Camera motion overwhelms scene motion.** A panning camera makes *everything* move. Global motion compensation (subtract the dominant camera motion first) is usually necessary.
- **Latency vs. compression tradeoff.** Higher compression = more residual = more tokens; low-bitrate streams paradoxically need *more* VLM tokens.
- **Interoperability.** Systems that receive already-decoded frames (RGB pixels) can't recover motion vectors without re-encoding. The trick is only worth it end-to-end from source stream.
- **Not a substitute for temporal modeling.** Fewer tokens per frame is still meaningless without attention that spans frames. Combine with temporal attention or a long-context language backbone.
- **Frame-sparse reasoning suffers if abused.** Aggressive suppression on "boring" static shots can drop tokens the model needed for reasoning about a still object. Tune density floors per region.

## Sources

- Paper: *Mage-VL: An Efficient Codec-Native Streaming Multimodal Foundation Model* — Microsoft Mage Team, 2026 — [arXiv:2607.24904](https://arxiv.org/abs/2607.24904).
- Related: standard H.264/H.265 codec references (Sullivan et al.); prior "motion-aware" video ViT tokenizers.
