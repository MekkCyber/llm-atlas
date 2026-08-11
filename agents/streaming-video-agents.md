# Streaming-video agents
*Depth — two-tier architectures for long-horizon interactive video understanding.*

**TL;DR:** Streaming multimodal agents must simultaneously respond to real-time queries and maintain hour-scale visual memory. Single-tier baselines all fail one axis: recent-frame agents lose distant events, text-caching agents lose visual evidence, memory-compressing agents lose fine detail. StreamMind's two-tier architecture assigns latency-critical interaction and proactive monitoring to independently-scheduled *frontend workers*, while *backend workers* asynchronously build persistent multimodal memory and perform historical recall / external search.

**Prereqs:** [README.md](README.md), [../multimodal/README.md](../multimodal/README.md)
**Related:** [../evaluation/streamarena.md](../evaluation/streamarena.md)

---

## What it is

An agent architecture that separates real-time reactive processing from long-horizon memory work, so the agent can respond to a query about the current frame in milliseconds while still being able to answer questions about events from many minutes ago.

## How it works

**Frontend workers (synchronous, latency-critical):**
- Ingest the audio-visual stream frame by frame.
- Handle live queries with a bounded recent-frame window.
- Perform proactive monitoring — detect noteworthy events and interject when appropriate.
- Independently scheduled per capability; parallel across frontend agents.

**Backend workers (asynchronous, throughput-oriented):**
- Consume the stream in parallel with the frontends.
- Build a **persistent multimodal memory**: chunk the stream into episodes, index visually and textually, retain scene traces at transitions.
- Handle historical-recall queries (e.g. "what was said at 0:14:22?") and external-search queries (retrieval-augmented over the persistent memory).
- Not on the response-latency critical path.

**Shared state.** The persistent memory is the interface between tiers. Frontends read from it when queries need history; backends write to it as new segments finalize.

**Query-to-answer latency** drops because frontends reuse prebuilt backend state rather than re-processing raw video per query.

## Why it matters

- **Simultaneously wins on multiple axes** — real-time perception, historical retrospection, proactive interaction, tool utilization — where single-tier baselines each blow one axis.
- **Latency budget stays bounded** even as video length grows, because heavy memory work is off the response path.
- **Composable memory structures.** The persistent memory can be per-episode fields (for temporal coherence) or verbatim landmark traces (for episodic recall) — see the parent paper's WorldTrace-style variants for the trade-offs.
- **Foundation for production interactive video assistants** (meeting bots, lecture-scale agents).

## Gotchas & tricks

- **Memory schema is load-bearing.** The exact structure of the persistent memory (what gets indexed, at what granularity) determines what queries the backend can answer cheaply.
- **Frontend-backend sync is a real engineering problem.** Stale memory when the frontend queries too early; contention when both tiers want the same GPU.
- **Cost scales with retention window.** Persistent memory grows with video length; require an eviction / compression policy.
- **Doesn't help clip-length benchmarks.** The architecture only pays off at hour-scale — StreamArena is the benchmark that exposes the wins.
- **Proactive interjection is a UX design axis, not just a technical one.** How often the agent volunteers information affects perceived quality more than raw QA accuracy.

## Sources

- Paper: *StreamArena: Toward Continuous, Interactive, and Long-Horizon Agentic Streaming Video Understanding* — Li, Zhu, Wang, Wu, Yu, Chu, Lu, Jia, Xiaohongshu / HKU / CUHK / HKUST, 2026 — arXiv:2608.05703. Introduces both the StreamArena benchmark and the StreamMind two-tier agent architecture.
