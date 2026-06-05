# Streaming spatial MLLM benchmark (OVO-S-Bench)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Most video-MLLM benchmarks are *offline* (full video given) and event-centric. OVO-S-Bench (Li et al., 2026) flips both: the model sees only the prefix preceding the query timestamp, and the questions probe *spatial* intelligence at four levels of abstraction — instantaneous perception, spatiotemporal tracking, spatial simulation, allocentric mapping. Headline finding: Gemini-3.1-Pro scores 59.2 vs. human 86.6, with allocentric mapping the dominant bottleneck.

**Prereqs:** [README.md](README.md), [../multimodal/README.md](../multimodal/README.md)
**Related:** [../post-training/reasoning/long-cot-rl.md](../post-training/reasoning/long-cot-rl.md)

---

## What it is

A benchmark for evaluating whether multimodal LLMs can reason about *space* (where things are, layout, viewpoint) from continuous egocentric video streams in the way an embodied agent (robot, AR headset, AV car) would have to. Two design choices distinguish it from prior video-MLLM benchmarks:

- **Streaming protocol.** Each question carries a query timestamp; at evaluation, the model sees *only the prefix up to that timestamp*. There's no peeking at future frames.
- **Spatial-first axis.** Existing benchmarks are dominated by events ("did X happen") and perception ("what color is the bag"). OVO-S-Bench targets spatial structure (layout, viewpoint, mapping).

## How it works

The 1,680 questions are human-annotated over 348 source videos — roughly 804 person-hours of annotation by 12 trained annotators each also serving as blind cross-reviewer. Each question has a query timestamp and an evidence interval (the frame range that contains the answer).

Questions are organized into four levels of increasing abstraction:

1. **Instantaneous egocentric perception.** What is visible right now?
2. **Spatiotemporal context tracking.** Where is something I saw earlier, given the camera has moved?
3. **Spatial simulation and reasoning.** If I rotate / step forward / look up, what will I see?
4. **Allocentric mapping.** Reason about the world in a map-frame independent of the camera (e.g. "is X north of Y").

At evaluation, the model receives the prefix prior to the timestamp and is scored on the answer.

## Why it matters

- **Streaming is the deployment regime.** Robotics, AR, autonomous driving — every embodied use of MLLMs is streaming. Offline benchmarks don't measure the real bottleneck.
- **Allocentric mapping is the next frontier.** The 27-point human gap (59.2 vs 86.6) is dominated by allocentric reasoning. This is a concrete, measurable target for the next round of training data and architectural work.
- **Negative results sharpen design.** Two strong negative findings: (1) streaming-fine-tuned and spatially fine-tuned MLLMs *underperform* their backbones, suggesting bad data or losses; (2) ungrounded chain-of-thought *amplifies* spatial errors. Both contradict received wisdom.

Across 38 proprietary and open-source MLLMs, current SOTA is well below human expert.

## Gotchas & tricks

- **Evidence-interval bias.** Models that condition on extra context (frames after the evidence interval but before the query) artificially do well; strict prefix-only is needed.
- **CoT carefully.** The "ungrounded CoT amplifies errors" finding implies prompts that *force* the model to reason in symbolic space without referring to the stream make things worse. CoT prompts that ground each step in the video help.
- **Annotator agreement.** Allocentric mapping questions are hard for humans too; the 86.6 human number is *expert*, not lay.
- **Training data implication.** The benchmark suggests current spatial training mixes are misaligned. Useful as a *diagnostic*, not just a leaderboard.

## Sources

- Paper: *OVO-S-Bench: A Hierarchical Benchmark for Streaming Spatial Intelligence in Multimodal LLMs* — Li et al., 2026 — [arXiv:2606.03890](https://arxiv.org/abs/2606.03890).
- Affiliation: InternLM.
