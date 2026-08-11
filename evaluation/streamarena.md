# StreamArena
*Depth — an hour-scale, open-ended, interactive streaming-video-agent benchmark.*

**TL;DR:** Streaming-video-agent benchmarks have suffered from two shortcuts: brief clips (so a "last 4 frames" minimal baseline wins) and multiple-choice format (so language shortcuts leak the answer). StreamArena fixes both — 243 full-length videos averaging **88.8 minutes**, and **3,646 open-ended QA pairs** exercising real-time perception, historical retrospection, proactive interaction, and multimodal tool utilization. The result exposes a tension between continuous interaction and long-horizon comprehension that prior benchmarks hid.

**Prereqs:** [README.md](README.md), [../agents/README.md](../agents/README.md), [../multimodal/README.md](../multimodal/README.md)
**Related:** [../agents/streaming-video-agents.md](../agents/streaming-video-agents.md)

---

## What it is

An evaluation for agentic systems that consume unbounded audio-visual streams and must maintain hour-scale memory. Unlike prior clip-level video-QA benchmarks, StreamArena imposes both real-time interaction (agent decides *when* to respond) and long-horizon evidence (answers depend on events tens of minutes earlier).

## How it works

**Corpus.**
- 243 videos, average length 88.8 minutes (hour-scale, not clip-scale).
- Full-length audio + video preserved — no pre-truncated windows.

**Question set.**
- 3,646 open-ended QA pairs (no multiple choice — no option leakage).
- Rigorously annotated across four capability axes:
  1. **Real-time perception.** What's happening *now* at query time.
  2. **Historical retrospection.** What happened many minutes ago.
  3. **Proactive interaction.** Agent-initiated interjection when something noteworthy occurs.
  4. **Multimodal tool utilization.** Using tools (search, retrieval) to answer.

**Evaluation protocol.** Agent processes the video as a live stream (not batch); queries arrive at annotated timestamps; open-ended answers scored by a strong judge model.

**Exposed tradeoffs (paper's key finding).** Baselines all fail differently:
- Recent-frame methods lose distant events (retrospection ≈ 0).
- Text-caching methods lose visual evidence (perception drops).
- Memory-compressing methods lose fine-grained detail (retrospection noisy).

## Why it matters

- **Kills the "last-4-frames wins" shortcut.** Hour-length videos + open-ended format + retrospection questions can't be aced by peeking at the recent frames.
- **Kills the multiple-choice shortcut.** Open-ended answers close the language-prior back-channel that MCQ leaves open.
- **Exposes an actual architectural tension.** No single existing streaming architecture wins across all four axes, motivating multi-tier designs (see [../agents/streaming-video-agents.md](../agents/streaming-video-agents.md)).
- **Reproducible measurement of "long-horizon multimodal memory."**  A capability everyone claims but few benchmarks measure honestly.

## Gotchas & tricks

- **Judge quality dominates open-ended scoring.** Report judge model and judge-vs-judge agreement.
- **Latency accounting.** "Real-time" is meaningless without a latency budget; StreamArena's proactive-interaction axis makes this explicit.
- **Compute cost is real.** Processing 88-minute videos per query is expensive to run at scale; sample per capability axis rather than exhaustively evaluating each system on the full set.
- **Retrospection annotation is painful.** Requires human annotators watching full videos; the 3,646 QA pairs represent substantial annotation effort.

## Sources

- Paper: *StreamArena: Toward Continuous, Interactive, and Long-Horizon Agentic Streaming Video Understanding* — Li, Zhu, Wang et al., Xiaohongshu / HKU / CUHK / HKUST, 2026 — arXiv:2608.05703.
