# Block-Causal Attention
*Depth — within-block bidirectional, across-block strictly causal masking for streaming foundation models.*

**TL;DR:** Slice the token stream into fixed-duration "blocks" (e.g. 160 ms of audio + the matching video tile). Inside one block, every token attends to every other — bidirectional, like an encoder. Across blocks, attention is strictly causal: a later block sees earlier blocks but never the reverse. The result is a single Transformer that can be both *streamable* (consume a new block the moment it arrives) and *contextual* (full bidirectional fusion within the block). Wan-Streamer's recipe for sub-second audio-visual response.

**Prereqs:** [attention](../fundamentals/attention.md), [multi-head-attention](multi-head-attention.md), [transformer-block](transformer-block.md)
**Related:** [mla](mla.md), [_normalization](_normalization.md)

---

## What it is

A modification of the attention mask. Standard causal attention forbids token *t* from looking at any *t' > t*. Block-causal attention groups consecutive tokens into a block $b$ and enforces:

```
mask(i, j) = 1 if block(j) <  block(i)        # earlier block — allowed
           | 1 if block(j) == block(i)        # same block    — allowed (bidirectional)
           | 0 if block(j) >  block(i)        # later block   — forbidden
```

A block is whatever unit makes streaming natural: 160 ms of audio frames, one video tile, one user-turn worth of text tokens.

## How it works

For Wan-Streamer the input sequence is a multimodal interleave — `[v_1 a_1 t_1 v_2 a_2 t_2 …]` — where each `(v_k, a_k, t_k)` triple is one streaming unit (≈160 ms at 25 fps). The model treats each triple as a single block:

- **Within a block.** Visual, audio, and text tokens attend bidirectionally. This is what lets the model fuse modalities — text can attend to audio that came in "after" it in the encoded sequence as long as both belong to the same time window.
- **Across blocks.** The mask is causal at the block level. Block $b$ can attend back to blocks $b-1, b-2, \ldots$, but not to $b+1$.

Output tokens (the model's response: TTS audio, talking-head video, text) are appended block-by-block at decode time. As soon as the encoder has finished one block of input, the decoder can produce one block of output — no waiting for "end of sentence."

Required co-design:

- **Causal encoders** (audio, video). The visual tokenizer cannot peek at future frames.
- **Causal decoders** (audio synth, video render). Same constraint in reverse.
- **Block-aligned positional encoding.** Either RoPE applied per-block-position, or learned block-id embeddings.

## Why it matters

- **Latency.** The model can emit its first response token after seeing one full block (~160 ms) of user input — not a full utterance. Wan-Streamer reports ~200 ms model-side latency end-to-end.
- **Single-model pipeline.** Removes the VAD → ASR → LLM → TTS → talking-head cascade. Error accumulation across modules disappears; cross-modal timing (laughs, interrupts, head nods) is learned jointly rather than scripted.
- **Drop-in compatible with FlashAttention.** The mask is still a triangular pattern — just a coarser one. No custom kernel needed.
- **Generalizes beyond speech.** Anywhere you have a real-time sensor stream (robotics, computer use, game agents) the same block-causal idea applies: bidirectional fusion within a step, causal evolution across steps.

## Gotchas & tricks

- **Block size is a latency / quality knob.** Smaller blocks lower latency but reduce within-block context, hurting prosody and multimodal alignment. 160 ms (4 frames @ 25 fps) is the sweet spot in Wan-Streamer.
- **KV cache management.** A block is added to the cache as a single chunk once finalized. This makes long-horizon attention very cheap (one KV entry per block, not per token).
- **Don't confuse with "sliding-window attention."** Sliding-window is still per-token causal; block-causal is bidirectional within the block. Different mask, different purpose.
- **Training requires masked-block losses.** Output tokens in a block must be trained to predict only from the inputs of the same and earlier blocks — careful loss masking is required to avoid teacher-forcing leakage.
- **Modality balance inside a block.** The interleave order (v then a then t? a then t then v?) matters at low precision. Empirically, ordering by sensor timestamp works best.

## Sources

- Paper: *Wan-Streamer v0.1: End-to-end Real-time Interactive Foundation Models* — Alibaba Wan team, 2026 — [arXiv 2606.25041](https://arxiv.org/abs/2606.25041).
- Related: streaming-transformer / chunked-attention designs for ASR and TTS, e.g. *Streaming Transformer ASR with Block-wise Synchronous Beam Search* (Tsunoo et al., 2020) — predecessor of the within-block-bidirectional idea in a single-modality setting.
