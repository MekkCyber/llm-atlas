# Speculative listening
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Full-duplex speech LMs spend most of their time listening to the user; during each listening chunk, they could be *drafting* the next audio chunk under the hypothesis that the user is silent. Speculative Listening does exactly that: draft one silence-continued chunk plus one chunk starting from the runner-up first token, verify when the real audio arrives, and emit accepted drafts with zero added latency. Lossless, training-free, model-agnostic. Introduced in 2611.06982.

**Prereqs:** [_speculative-decoding](_speculative-decoding.md)
**Related:** none

---

## What it is

An inference-side scheduler that repurposes idle *listening* time in a chunk-interleaved full-duplex speech LM. Full-duplex speech models alternate between listen-chunk (user audio comes in, model attends) and speak-chunk (model emits audio). During a listen chunk, the model's compute is used to attend to user audio but is otherwise idle for the speak pathway. Speculative Listening fills that idle budget with speculative drafts for the next speak chunk.

## How it works

Let $C_t$ denote the chunk boundary at time $t$. The model is listening over $[C_{t-1}, C_t]$; the real audio for that interval is only known once $C_t$ arrives.

1. **Silence-branch draft.** Assume the user stays silent for $[C_{t-1}, C_t]$; under that assumption, decode the next *speak chunk* starting at $C_t$. This is the "most likely continuation if nothing changes" branch.

2. **Runner-up branch draft.** The model's prediction for the first token of the next speak chunk has a top-1 and a runner-up. Spawn a second decode starting from the runner-up first token, so that if the real audio invalidates the silence branch, the runner-up branch is already partially decoded.

3. **Verification.** When $C_t$ arrives, run one forward pass on the real audio chunk + the drafts. Accept the branch whose first-token prediction matches; discard the other.

4. **Emission.** Accepted drafts emit immediately — pre-decode latency is zero for the accepted branch.

The scheme is lossless (the acceptance rule is argmax of the base distribution) and training-free (no auxiliary drafter model).

## Why it matters

- **Audio-stream analogue of EAGLE/Medusa.** Parallelizes predictive work with verification, but the parallelism axis is *idle listening time* rather than lookahead.
- **Latency budget for voice agents.** First-audible-token latency is the bottleneck for real-time voice; cutting the pre-response decoding delay by 73% on average is meaningful.
- Reported speedups: **2.6–2.8× on speak chunks**, up to **23.4× on tool-call chunks** (where the response is highly predictable), **73% average reduction** in pre-response decoding delay on the paper's TickTock-Bench suite.

## Gotchas & tricks

- The method relies on the model being chunk-interleaved — not all full-duplex speech LMs have this structure.
- Two decoding branches use extra KV cache; on memory-limited devices, the runner-up branch can be dropped at the cost of some acceptance rate.
- The win on tool-call chunks is so large (23.4×) because tool responses have highly predictable boilerplate; variable-content utterances see the smaller 2.6–2.8× gain.
- Doesn't help on first-ever response (no previous listen-chunk to draft during) — the latency of the opening response is unchanged.

## Sources

- Paper: Decode Before You Hear: Speculative Listening for Chunk-Interleaved Full-Duplex Speech Language Models — Bian et al., 2026 — [arXiv:2611.06982](https://arxiv.org/abs/2611.06982)
