# Speculative listening
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Full-duplex speech language models process audio in fixed time chunks; the "listen" chunks leave compute mostly idle. Speculative Listening (2026) fills that idle time by **drafting the next chunk under an assumed input (silence)**, then verifying against the real audio when it arrives. Training-free, lossless, and yields up to 23.4× speedup on tool-call chunks.

**Prereqs:** [speculative-decoding](speculative-decoding.md)
**Related:** none

---

## What it is

Chunk-interleaved full-duplex speech LMs alternate between short "listen" chunks (consume incoming audio) and "speak" / "tool-call" chunks (generate audio or structured outputs). Each chunk must complete before the next arrives. Workload is lopsided: listen chunks leave the GPU underused, speak chunks are compute-bound and serial.

Speculative Listening is a speech-LM-native variant of speculative decoding that uses listening idle compute as the drafting budget.

## The two observations that make it work

The paper measures, across four full-duplex speech models:

1. **95.3% of chunks:** the next audio chunk does *not* change the output. Decoding under assumed silence gives the correct output.
2. **88% of changes:** the divergence begins at the **first output token** of the chunk.

So the vast majority of speculated outputs will be accepted verbatim; of the 4.7% that will be revised, nearly all are fixable by having a second branch that starts from the runner-up first token.

## How it works

**During idle time** (while listening for the next chunk):
- **Main branch.** Speculate the next chunk's output under the assumption that the incoming audio is silence.
- **Second branch.** From the runner-up for the first output token, continue speculating. This cheaply covers the "first-token change" case.

Both drafts go into a **cross-chunk hypothesis pool**.

**When the real audio arrives:**
- Run the model one step with the real audio to verify the first token.
- If the first token matches the main branch, commit it and continue verifying the rest of the main branch.
- If the first token matches the second branch, switch and verify the second branch.
- If neither, fall back to standard decoding for that chunk.

The verification is a single forward pass; acceptance is exact (same distribution as the non-speculative run).

## Why it matters

Interactive speech systems live or die by **time-to-first-response**. Previous optimisations (quantization, KV caching, smaller models) trade quality for latency. Speculative Listening is lossless and uses the structural slack already present in the full-duplex schedule.

Reported numbers on TickTock-Bench: speak-chunk speedup 2.6–2.8×; tool-call-chunk speedup up to 23.4×; average decoding delay cut by 73% to 74–241 ms — close to the ~200 ms human turn-gap — and maximum real-time lag reduced by 94.4%.

## Gotchas & tricks

- **Silence assumption** is domain-specific. For a very noisy environment the under-silence draft diverges more often; adaptive draft input (e.g. last chunk's audio replayed) is a natural extension.
- **Second branch cost.** Running two drafts doubles the idle-time budget but still fits inside the listen chunk on measured workloads.
- **Hypothesis pool management.** Across multiple chunks of drift, the pool must be pruned to avoid stale drafts being verified against audio they no longer apply to.

## Sources

- Paper: Decode Before You Hear: Speculative Listening for Chunk-Interleaved Full-Duplex Speech Language Models — Bian et al., 2026 — [arXiv:2611.06982](https://arxiv.org/abs/2611.06982)
