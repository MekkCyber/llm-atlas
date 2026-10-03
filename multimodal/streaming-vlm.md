# Streaming Video LLMs
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A VLM that consumes video as it arrives, instead of over a fixed, finite clip. The system must decide (a) *what evidence to retain* before any question is known, and (b) *when to respond* once evidence is sufficient — jointly, under tight compute. Modern designs (OneStreamer, 2026) frame perception, memory formation and timing as a single proactive-generation process so the three objectives share a backbone.

**Prereqs:** [../architectures/transformer-block.md](../architectures/transformer-block.md)
**Related:** [../agents/README.md](../agents/README.md) · [../inference/README.md](../inference/README.md)

---

## What it is

An offline video LLM gets a whole clip and a question together, and can attend over the entire visual context. A streaming VLM gets frames one at a time and must emit outputs (QA, captions, alerts, grounded moments) *during* the stream, without rewinding and without unbounded context growth. Two sub-problems make this hard:

1. **Query-independent memory.** Future questions are unknown, so what the model saves of the past has to be useful for queries that may never be asked.
2. **Timing.** The model must choose when to speak; always-on commentary floods bandwidth, waiting-only kills responsiveness.

## How it works

Three recurring design elements:

1. **Short visual window.** A sliding window of raw frames (seconds to tens of seconds) for real-time perception.
2. **Generative long-term memory.** Older frames are *replaced* by model-written artifacts — timestamped captions, event summaries, scene graphs — that persist after their frames leave the window. OneStreamer's Proactive Hierarchical Caption Memory (PHCM) is the current public reference: local-detail captions per short interval + summaries of completed events.
3. **Learned timing head.** A token-level supervision signal distinguishes *state-change* and *state-persistence* positions from the dominant "waiting" state, so the model learns when a response is due. OneStreamer's Proactive State Transition Learning (PSTL) supervises ~27.5% of state tokens and beats dense supervision.

Training couples these: streaming caption targets supervise what to save; QA targets supervise how to use saved memory; state targets supervise when to speak.

## Why it matters

- **Decouples retention from attention span.** A fixed visual window + generative memory lets context grow with text instead of frames; this is the only known way to run multi-hour sessions on realistic hardware.
- **Proactive over reactive.** Streaming agents (interview assistants, surveillance, driving copilots, live tutoring) need to volunteer information; a reactive VLM that only answers prompts is the wrong product.
- **A common interface** across perception, memory and response means one loss landscape rather than three stitched-together models.

## Gotchas & tricks

- Written memory can hallucinate; validate against the window before committing to a time-grounded claim.
- "Waiting" dominates streaming traces — straight next-token loss drowns out timing-relevant tokens. Up-weighting state tokens (PSTL) is a cheap fix.
- Benchmark coverage is thin: most video QA sets assume full-clip access and don't test memory decay or timing. Dedicated streaming benchmarks are still new.
- Visual-window size is a sharp tradeoff: larger → better local perception, more compute per frame; smaller → leaner but more pressure on generative memory accuracy.

## Sources

- Paper: *OneStreamer: Unifying Perception, Memory, and Proactive Response in Streaming Video Interaction* — Zeng et al., Nanjing University, 2026 — [arXiv:2610.01762](https://arxiv.org/abs/2610.01762).
- Related: VideoLLM-online, Flash-VStream (earlier streaming VLMs with simpler memory).
