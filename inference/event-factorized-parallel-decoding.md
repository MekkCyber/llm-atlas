# Event-factorized parallel decoding
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** For structured outputs like dense video captions (many events over long video), token-level dependencies are *weak across events but strong within events*. Restructure the causal graph: decode within-event tokens sequentially (preserves local coherence), but decode across-event tokens **in parallel** (they're near-independent). Made lossless by a **latent global planner** that emits compact event-structure tokens before decoding — the planner encodes inter-event causality so parallel per-event decoding still respects global structure.

**Prereqs:** [../multimodal/README.md](../multimodal/README.md)
**Related:** [_speculative-decoding.md](./_speculative-decoding.md)

---

## What it is

Autoregressive video LMs decode dense captions one token at a time, which scales badly as video length and event density grow. But dense captioning has an unusual structural property: the tokens describing event A ("A dog runs across the frame from 0:03 to 0:07") are largely independent of the tokens describing event C ("A cat jumps out of frame at 1:22"), even though a full AR model treats every prior token as causal context.

Event-factorized parallel decoding exploits this weak-cross-event / strong-within-event dependency pattern to decode multiple events in parallel while keeping token-order coherent within each event.

## How it works

Two components:

**1. Latent global planning.** Before token decoding, run a lightweight planner that produces compact **global planning tokens**. Each planning token summarizes an event's semantic content (from audio-visual features) and its position in the inter-event causal chain (temporal order, referential dependencies). These tokens are what makes the decomposition lossless — the parallel per-event decoders condition on the *same* set of planning tokens, so global consistency is preserved.

**2. Event-factorized parallel decoding.** With planning tokens fixed, the token stream is factored into per-event token groups. Each event's tokens are decoded sequentially (preserving local semantic flow: subject-verb-object ordering, tense agreement). But different events' token groups decode **in parallel** on the same forward pass. Cross-event attention still exists (through the planning tokens) but not through per-token dependencies.

The result: wall-clock decoding time roughly proportional to the *longest* single event's caption length, not the sum of all events.

## Why it matters

- **Better efficiency *and* accuracy** vs sequential AR on omni-modal dense video captioning benchmarks — the planner's inter-event awareness produces cleaner global structure than plain AR, and parallelism cuts wall clock.
- **Generalizes the parallel-decoding insight** beyond speculative decoding. Speculative decoding exploits *low-difficulty* tokens; event-factorization exploits *structural independence*. Both make more of the output "free."
- **Transferable pattern.** Any domain where the output has natural segmentation (multi-turn dialog, multi-file code generation, multi-section report writing) is a candidate for the same treatment.

## Gotchas & tricks

- **Planner quality is the bottleneck.** If the planner misidentifies event boundaries or misses cross-event dependencies, parallel decoding produces inconsistent captions (event C's tokens contradict event A's). Training the planner well is the whole game.
- **Attention pattern is non-trivial.** Not a standard causal mask — you need block-diagonal per-event masking with global attention to planning tokens. Requires custom kernel or masked-attention fallback.
- **Not lossless in general** — it's lossless *given the planner captures the relevant inter-event causality*. Domains where cross-event token-level dependencies matter (poetry, tight prose) don't fit.
- **Distinct from speculative decoding.** No drafter, no verifier — it's a re-factoring of the AR generation itself. The planning tokens are trained end-to-end with the LM loss.

## Sources

- Paper: *Parallelized Autoregressive Decoding for Omni-Modal Dense Video Captioning* — Jiao, Gao, Ng, Shou, National University of Singapore, 2026 — [arXiv:2607.02963](https://arxiv.org/abs/2607.02963).
