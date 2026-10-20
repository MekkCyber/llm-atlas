# History-Free Speculative Drafter (DLite)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A speculative-decoding drafter built for **agentic workloads** — long, evolving contexts that break conventional draft models. Instead of keeping its own KV cache, the drafter **reads the target model's hidden states directly** and predicts the next tokens locally. No draft history to maintain, no cache to flush when the context changes, acceptance stays high as the agent loop evolves.

**Prereqs:** [attention.md](../fundamentals/attention.md)
**Related:** _(none yet in graph — this folder is being seeded)_

---

## What it is

Speculative decoding uses a small **draft model** to propose several tokens that a large **target model** verifies in one pass. In standard setups the drafter is a smaller LM with its own KV cache: that cache must be maintained across turns and tracks the full prefix. In agent loops — multi-turn tool calls, long tool outputs, context evolving faster than the drafter can keep up — this cache both costs memory and *goes stale*: the drafter's representation of history diverges from the target's. Acceptance length drops, and the speedup collapses.

**DLite** removes the drafter's own history entirely. At each draft step, a small head reads the target model's **committed hidden states** (which already encode history) and predicts the next tokens locally.

## How it works

**Shared target hidden state.** When the target model commits a token, its last-layer hidden state is already available (the model just produced it). DLite reuses that state as the drafter's input signal.

**Local predictor.** A lightweight head — a few transformer-like layers that attend over a small local window of target hidden states, not back to a persistent cache — produces the next $k$ draft tokens. Because its only input is the target's current state, there is no separate KV to maintain.

**Standard verification.** The target verifies the $k$-token draft in one forward pass as in vanilla speculative decoding; accepted tokens commit, rejected ones trigger a re-draft at the correction point.

**No cross-turn state.** When the agent loop appends a tool output and reruns the target, DLite starts from the target's new hidden state. There is no stale drafter cache to invalidate.

## Why it matters

- **Agentic workloads.** This is where speculative decoding has struggled most. By removing drafter state, DLite keeps acceptance length high across tool calls and context mutations where conventional drafters degrade.
- **Memory footprint.** No draft KV cache per request — relevant for high-concurrency agent serving.
- **No per-workload retraining.** The predictor head trains once alongside the target; it does not need to specialize to a specific agent trace distribution.

## Gotchas & tricks

- **Acceptance length is bounded by local predictability.** With no draft history, DLite cannot speculatively look back further than the target's own state exposes. For non-agent workloads (long single-document generation), a conventional drafter with full history can match or beat it.
- **Target hidden-state access.** The drafter has to read the target's last-layer state, so the serving stack must expose that output (vLLM/SGLang both do, but some custom inference stacks don't).
- **Compatibility with KV-cache tricks.** Paged attention and prefix caching on the target are orthogonal and compose cleanly.

## Sources

- Paper: *DLite: History-Free Speculative Decoding for Agentic AI* — Wang et al., 2026 — [arXiv:2610.23787](https://arxiv.org/abs/2610.23787)
