# Codebook-MTP Audio Decoder
*Depth — a multi-token-prediction audio decoder with shared early layers, per-codebook late layers, and partial-fusion feedback to the backbone.*

**TL;DR:** Residual-codec audio generators have to emit many parallel codebook streams per frame. The naïve designs either flatten codebooks into a long sequence (slow) or predict all codebooks from a single hidden state (loses per-codebook capacity). The **Fish Audio Drama 3** decoder splits the difference: **4 shared layers** build a frame-level representation from the backbone, then **9 per-codebook layers** (one head per residual codebook) refine it. To let semantic structure flow back into the backbone without choking the decoder, only the two lowest codebooks (`c_0` semantic + `c_1`) are summed back into the backbone, scaled by `1/√2`. The design preserves per-codebook capacity, enables streaming, and keeps the backbone light.

**Prereqs:** [../pre-training/mtp.md](../pre-training/mtp.md), [../fundamentals/attention.md](../fundamentals/attention.md)
**Related:** [../architectures/_moe.md](../architectures/_moe.md) · [../case-studies/fish-audio-drama-3.md](../case-studies/fish-audio-drama-3.md)

---

## What it is

An audio-side decoder for a residual-codec speech LM. The backbone emits one hidden state per audio frame; the decoder turns that state into the frame's multiple codebook tokens (10 codebooks per frame in Drama 3, 1 semantic + 9 residual acoustic). The decoder is "MTP-shaped": many tokens predicted at the same position, but with a structured per-codebook pathway rather than a single wide head.

## How it works

**Three structural pieces.**

1. **Shared layers (×4).** Standard transformer blocks (gated full attention + SwiGLU FFN at backbone width) that take the backbone's per-frame hidden state and produce a frame-level representation. All codebooks share these.
2. **Per-codebook layers (×9).** One block per residual codebook, each with its *own* weights and output head. The input to codebook `c_i`'s block is the previous layer's output *plus* the embedding of the already-predicted lower codebooks `c_0, …, c_{i−1}` for the same frame. Each block attends causally across the full sequence (not just within a frame).
3. **Partial fusion back to the backbone.** The predicted embeddings of `c_0` (semantic) and `c_1` are **summed back into the backbone** at the next frame, scaled by `1/√2`. Codes `c_2, …, c_9` do not feed back — they flow only to the decoder.

**Why partial fusion.** The semantic token `c_0` carries most of the linguistic content and must be visible to the backbone to track phonetics across frames. `c_1` adds coarse prosody. The higher-residual codebooks mostly carry fine acoustic detail; feeding them back would widen the backbone's effective input without a corresponding gain in text-level planning. The `1/√2` scale is an equal-variance balance.

**Streaming behavior.** Because every layer attends causally across the whole sequence, the decoder streams: as soon as the backbone emits a frame's hidden state, the shared layers run, then the 9 per-codebook heads run in sequence within that frame, and the codec can decode immediately.

**Training signal.** A per-codebook cross-entropy loss. In Drama 3's loss, these are averaged into one *codebook loss* with coefficient 0.5, added to a 0.5 *semantic loss* on `c_0` and a 1.0 *text* cross-entropy on the backbone's text tokens.

## Why it matters

- Keeps the backbone *small* (a 35B MoE with 3B active) by not forcing it to carry every codebook's state across the whole audio stream — the decoder handles the fine-grained acoustic prediction.
- Preserves per-codebook capacity (9 separate heads with their own weights) without flattening codebooks into a long sequence that would multiply sequence length by 10×.
- Streaming-friendly: no cross-frame batched decoding tricks required.
- Enables partial-fusion as a tunable *abstraction level*: how much acoustic detail the backbone sees is a design knob, not an all-or-nothing choice.

## Gotchas & tricks

- **Fusion scale `1/√2`.** For two summed signals of unit variance, the sum has variance 2; dividing by √2 restores unit variance. Important for optimization stability early in training.
- **Fusion depth is a design knob.** Feeding back only `c_0` is cleaner but loses coarse prosody; feeding back `c_0..c_3` widens the backbone and slows it. Drama 3 picks `c_0..c_1`.
- **Per-codebook layer ordering matters.** The lower residual codebooks must be predicted *before* the higher ones use them as context; the paper's order is strictly `c_0 → c_9`.
- **Not interchangeable with "flat" MTP.** This design is specific to residual-codec audio; applying the same block to text multi-token prediction (DeepSeek-V3 style) doesn't match the semantics.
- **Freezing in RL.** During RL fine-tuning, Drama 3 **freezes the decoder** and trains only the backbone. The decoder is a converged acoustic model; backbone-only RL changes the planning distribution without re-opening acoustic quality.

## Sources

- Paper: *Fish Audio Drama 3 Technical Report* — Fish Audio, 2026 — arXiv:2610.18010.
