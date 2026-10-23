# Agent KV-Cache (Structured, Editable)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** When an LLM agent accumulates long interleaved observation/action trajectories, naive KV-cache management breaks: trimming or summarizing the *beginning* of the trace invalidates every cached key/value *after* the edit, forcing a full prefill. Agent KV-caches instead treat the cache as a structured data store with explicit edit regimes — prefix-anchored summaries, rolling suffix buffers, offloaded blocks — chosen so edits touch bounded portions of the trace.

**Prereqs:** [attention](../fundamentals/attention.md), [mla](../architectures/mla.md)
**Related:** [transformer-block](../architectures/transformer-block.md)

---

## What it is

In a standard decoder serving stack (vLLM, SGLang), the KV cache is append-only: each new token appends per-layer K/V vectors, and attention at position *t* reads positions `0..t`. This is fine for chat because the context never changes once written.

Agents violate that assumption. A long-horizon agent trajectory interleaves observations, tool outputs, and actions over hundreds of turns; the active context must be bounded. The usual answer is to compress or drop *older* turns — but any mutation to position *k* invalidates every K/V at positions `>k`, because those keys were computed with full attention over positions `<k`. A single edit at turn 20 of a 200-turn trajectory means re-prefilling turns 21–200.

Agent KV-cache designs recover edit-locality by changing *how* the cache is structured, not *what* the model computes.

---

## How it works

Two orthogonal design axes:

**1. What gets kept.** Options:
- **Raw tokens** — accurate but grows unbounded.
- **Summarized spans** — a short natural-language summary written back as tokens; cheap, lossy, re-encoded on each summary refresh.
- **Learned soft tokens** — a small bundle of continuous vectors trained to recover the span's downstream effect; dense, requires training.
- **Offloaded blocks** — K/V tiles moved to CPU/NVMe and streamed back on access; lossless, I/O-bound.

**2. Where edits are allowed.**
- **Prefix-anchored** — a fixed, immutable "scratchpad" prefix (system prompt, running summary) is never edited; all mutations happen in the suffix. Preserves prefix caching entirely.
- **Suffix-only rolling** — a fixed-size window near the end moves forward; everything older gets folded into the prefix summary once.
- **Arbitrary edits with recomputation masks** — edits allowed anywhere; a recomputation mask tells the serving layer the minimal set of positions whose K/V needs refresh.

The practical sweet spot combines prefix-anchored summaries with a rolling suffix: the prefix summary is updated only at session boundaries (so prefix caching stays warm across turns), and the rolling window captures the near-term observation/action trace.

---

## Why it matters

End-to-end agent throughput is dominated by prefill-after-edit cost, not model forward passes. Standard summarizers appear to compress context but trigger a full re-prefill every time the summary rewrites. Structured edit regimes let the serving stack cache prefix K/V across turns while the agent still appears to have an "evolving" context. This is a prerequisite for serving long-horizon agents at interactive latency without massive KV budgets per sequence.

---

## Gotchas & tricks

- **Summary refresh triggers a prefill.** Even "prefix-anchored" designs still pay when the summary itself is rewritten. Amortize by batching summary updates across many turns.
- **Rolling window + attention sinks.** Pure rolling windows drop early tokens that often serve as attention sinks; keeping the first *k* tokens (StreamingLLM-style) alongside the window usually recovers quality.
- **Soft tokens need task-specific training.** Attractive on paper but adds a training step that most agent frameworks skip.
- **Measure under real agent workloads.** Chat-style KV benchmarks don't exhibit the edit pattern that breaks naive caches; use trajectories with many short turns, not few long turns.

---

## Sources

- Paper: *Cache Engineering for Agents* — Tianyu Fu, Ruoxi Wang, Zihan Min, et al., Tsinghua / CUHK, 2026 — [arXiv:2610.28577](https://arxiv.org/abs/2610.28577).
- Background: *StreamingLLM* (attention-sink-style rolling windows); vLLM / SGLang prefix-caching documentation.
