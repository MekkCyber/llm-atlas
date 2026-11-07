# Speculative Listening
*Depth — speculative decoding specialized to full-duplex speech LMs, using idle "listen" compute to pre-draft the next "speak" chunk.*

**TL;DR:** Full-duplex speech LMs alternate **listen** (take audio in) and **speak** (emit audio out) chunks; during listen, the big model is idle. **Speculative Listening** uses that idle compute to pre-draft the *next* speak chunk under a silence assumption plus a runner-up-first-token branch, then verifies when real audio actually arrives. Training-free, lossless, and reports 2.6–2.8× speedup on speak chunks, up to 23.4× on tool-call chunks, a 73% reduction in pre-response delay, and a 94.4% reduction in maximum lag behind real time.

**Prereqs:** [speculative-decoding](./speculative-decoding.md)
**Related:** [speculative-decoding](./speculative-decoding.md)

---

## What it is

Chunk-interleaved full-duplex speech LMs process audio in fixed-length chunks, alternating "listen" and "speak" roles. During a listen chunk the model only *receives* audio; its KV is updated, but it emits nothing — all its compute during that chunk is otherwise wasted if nothing can be done with the output yet. Speculative Listening treats the listen chunk as the drafting phase of standard [speculative-decoding](./speculative-decoding.md): the big model drafts its own response, speculatively, under a specific *hypothesis about the user's next audio*.

## How it works

Three moves:

1. **Silence hypothesis.** Assume the next listen chunk is silence (or any audio the current KV already implies the user is not going to speak further). Generate speak-chunk tokens on that hypothesis during what would have been idle time.
2. **Runner-up first-token branch.** The common "user interrupts with a short reply" case is covered by a second branch: fork the KV at the chunk boundary; one branch assumes the user is quiet, the other assumes the user said the predicted runner-up first token (e.g. "yes"/"no"). Keep both branches drafted.
3. **Verification on arrival.** When the real audio arrives, do one target-model pass on the real listen chunk, and compare the resulting KV + first speak logits against the drafts. The branch (if any) that matches is accepted and immediately continues speaking; mismatches are discarded and the model falls back to normal decoding.

Because the model that drafts and the model that verifies is the same model, there is no drafter to train and no distribution mismatch: the draft is lossless by construction once the audio hypothesis is confirmed.

## Why it matters

- **Latency is the dominant UX cost of full-duplex speech.** Users feel ~200 ms as "slow." Pre-response delay was a structural lower bound tied to the chunk boundary; speculative listening shortens it to the mismatch-detection cost.
- **Idle compute is free.** In a fixed-size listen chunk, the big model is memory-bound anyway; running a draft during that window costs no additional GPU-seconds.
- **Scales with tool-use.** Tool-call chunks are highly predictable (the model has already decided to call a tool), so the silence hypothesis almost always wins; the 23.4× speedup on tool calls comes from this.

## Gotchas & tricks

- **Branch budget.** More hypotheses → more drafted tokens → more rollback cost on mismatch. Two branches (silence + runner-up) is empirically the sweet spot.
- **Not quite lossless at the boundary.** If the draft is accepted and the first actual audio timestamp causes a tiny KV shift that would have changed logits, you can get a one-token distributional drift. The paper reports this is below audible threshold but it is a measurable artifact.
- **Interaction with turn-taking logic.** The speech LM has to decide *whether to speak at all* under the silence hypothesis. Overly eager drafting results in the model talking over the user; a conservative "speak-start" threshold mitigates this.
- **Chunk size matters.** Longer listen chunks → more draft runway → bigger speedup, but worse turn-taking responsiveness. The paper tunes to the Moshi-style ~80 ms chunk.

## Sources

- Paper: *Decode Before You Hear: Speculative Listening for Chunk-Interleaved Full-Duplex Speech Language Models* — Bian et al., CUHK-SZ, 2026 — [arXiv:2611.06982](https://arxiv.org/abs/2611.06982).
- Background: *Fast Inference from Transformers via Speculative Decoding* — Leviathan et al., 2023 — see [speculative-decoding](./speculative-decoding.md).
- Background: *Moshi* — Kyutai, 2024 — the canonical chunk-interleaved full-duplex speech LM architecture.
