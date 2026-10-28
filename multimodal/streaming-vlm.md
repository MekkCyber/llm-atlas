# Streaming VLMs — proactive, timing-aware video assistants
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A **streaming VLM** continuously watches a video and decides not only *what* to say but *when* to say it. Classical VLMs answer when asked; a streaming VLM infers implicit user intent from the ongoing scene, maintains a hierarchical memory of evolving intent, and emits responses only when they would help. StreamVibe-VL (2026) ships one such model plus a benchmark (StreamVibe-Bench, 4.1K eval / 85K train samples) and reports **+10.7%** on combined response-timing/answer-quality and **+12.3%** on proactive interactions over strong streaming baselines.

**Prereqs:** [../multimodal/README.md](../multimodal/README.md)
**Related:** [../agents/README.md](../agents/README.md), [../evaluation/README.md](../evaluation/README.md)

---

## What it is

Three capabilities differentiate a streaming VLM from a classical VLM:

1. **Online perception.** The model consumes video frame-by-frame (or chunk-by-chunk) with bounded compute per step; it cannot re-watch the entire video to answer a query.
2. **Response-timing head.** A per-step classifier decides whether now is the right moment to produce an answer. The space of outputs at each step is {remain silent, speak}.
3. **Evolving-intent memory.** User intent is not fixed at prompt time; it drifts as the scene unfolds. A hierarchical memory (short-term scene buffer, long-term intent log) tracks that drift and conditions both the "speak now?" head and the generation head.

## How it works

- **Per-frame feature stream.** Visual encoder produces frame tokens at a steady rate; frames are pooled into short-window representations by the short-term memory.
- **Intent inference.** A dedicated head reads short-term memory + the long-term intent log and updates an estimate of the user's current goal.
- **Timing decision.** A separate head reads the intent estimate and current scene state and produces a binary "respond now" signal; thresholding is calibrated per task.
- **Response generation.** When the timing head fires, the model generates a response conditioned on the full memory.
- **Harness glue.** StreamVibe's harness manages memory decay, intent updates on drift, and interruption handling (user speaking over the model).

## Why it matters

- **"When to speak" is the gating skill for live video copilots** — AR glasses, continuous meeting assistants, driving/sports coaches. Classical VLMs are structurally wrong for this because they're trained to answer when asked.
- **Timing is measurable now.** StreamVibe-Bench provides a public eval that scores *both* answer quality and response timing, which turns proactive-video from a vibe-based pitch into a benchmarked capability.
- **Hierarchical memory + "speak now?" is a reusable pattern** for other streaming modalities (continuous audio, screen-sharing agents).

## Gotchas & tricks

- **Threshold calibration is domain-specific.** A model well-tuned for "coach" tasks is likely trigger-happy for "silent assistant" tasks and vice versa.
- **Latency budget caps model size.** Running a VLM on every frame is expensive; most useful streaming VLMs will be mid-sized models with aggressive KV-reuse.
- **Benchmark gaming on timing is possible.** A model that waits for the user to finish before answering can get decent answer-quality numbers while being useless as a proactive assistant — the paper explicitly scores both dimensions.

## Sources

- Paper: *StreamVibe: Proactive Video Agents for Evolving User Intent* — Liu et al., Google and co-authors, 2026 — [arXiv:2610.34662](https://arxiv.org/abs/2610.34662).
