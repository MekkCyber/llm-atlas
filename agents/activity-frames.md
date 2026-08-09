# Activity Frames

*Depth — deterministic, model-free compilation of raw screen-capture streams into typed episodes for computer-use agent memory.*

**TL;DR:** Computer-use agents keep re-deriving routines the user already performed because their memory records what the user *said*, not what they *did*. Activity Frames (2026) is a deterministic, zero-model pipeline that segments passive screen capture into typed "activity frames" — bounded episodes carrying application, site, timing, input volume, and evidence pointers back to the raw capture. Because no LLM is in the compilation loop, the output is byte-identical, cacheable, and mechanically auditable, and empirically it beats LLM summarization both on cost and on downstream agent QA accuracy.

**Prereqs:** *(none — this is the raw-signal side of the agent-memory stack)*
**Related:** [../post-training/vlm-as-judge.md](../post-training/vlm-as-judge.md)

---

## What it is

A compiler that takes a **local capture stream** (screenshots + input events + window metadata) and emits a canonical stream of **activity frames**. Each frame is a typed record:

```
ActivityFrame {
  application, site
  time_start, time_end
  input_volume    // keystrokes, mouse events, scroll distance
  evidence_ptr    // byte range in the raw capture (for audit / replay)
  type            // e.g. read, write, browse, meeting, terminal
}
```

Downstream, the compiled block plugs into the agent's context as prompt-ready memory: "what the user did in the last N hours, in structured form the agent can query."

---

## How it works

1. **Passive capture.** Continuously record screen frames, keystrokes, mouse events, and application metadata locally.
2. **Segmentation.** Deterministic rules break the stream into candidate episode boundaries (application-switch, long idle, focus loss, URL change, sustained input pattern change).
3. **Typed compilation.** Each episode is annotated with fixed-schema fields — no free-text summary. Field values are computed by rules or lookups over the raw capture.
4. **Evidence pointers.** Every frame keeps a byte offset into the raw stream so anything downstream can drill into the source. Cacheable because the same input yields the same output byte-for-byte.
5. **Consumption.** The agent reads the compiled block as compact context and, when it needs detail, dereferences an evidence pointer.

## Why it matters

- **Byte-identical outputs mean cacheable memory.** LLM summaries drift across runs; deterministic compilation doesn't. Memory that never drifts is trivially cache-shareable across agent instances and inspectable by the user.
- **Massive compression at low latency.** Reported: 86× smaller than raw capture, 68 ms per day of data, on a 128k-frame single-user corpus.
- **Better downstream QA than LLM summary.** 98.4% accuracy (Wilson 91.7–99.7%) for a downstream agent reading the compiled block, vs. 66–80% for the same agent reading an LLM summary of the same raw capture. A mid-tier model on the compiled block matches a frontier model on the raw capture.
- **Auditability and privacy.** No opaque LLM step; users can inspect (and redact) the compiled block deterministically.

## Gotchas & tricks

- **The schema is the ceiling.** Fields chosen by the compiler bound what the downstream agent can ever answer. Extending the schema is cheap (add a field, recompile) but requires a schema change.
- **Segmentation heuristics dominate quality.** Where the compiler places episode boundaries determines how legible the block is. Bad heuristics fragment a single "reviewing a PR" episode into 30 micro-episodes.
- **Not a substitute for content understanding.** Activity frames answer "when and where," not "what does the doc *say*." For content, downstream agents still need to open evidence pointers and read.
- **Evaluation is single-user for now.** Reported metrics come from one user's 128k-frame corpus; generalization across users, OSes, and application ecosystems is not yet quantified.
- **Complements, doesn't replace, learned memory.** For semantic recall ("that thing I read about GRPO last week"), a learned retrieval layer over the raw capture is still needed. Activity frames are the *skeleton*, not the whole memory.

## Sources

- Paper: *Activity Frames: Deterministic Screen-Activity Compilation for Agent Memory and Replay* — Anonymous, 2026 — [arXiv:2608.05784](https://arxiv.org/abs/2608.05784).
