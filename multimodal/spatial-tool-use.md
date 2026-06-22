# Spatial Tool Use for VLM Reasoning
*Depth — VLMs that call 3D-aware tools mid-thought and maintain a persistent scene state across calls.*

**TL;DR:** VLMs reason from static visual observations and lose spatial information across turns. **S-Agent** (Dai et al., NTU + Tsinghua + ByteDance, 2026) binds a VLM to a **spatial tool library** (depth estimation, segmentation, 3D scene construction, view projection) and interleaves tool calls with chain-of-thought. Each tool call updates a **shared scene context** that downstream reasoning can read. Turns a static-image VLM into a stateful spatial reasoner over continuous multi-view images and video.

**Prereqs:** [../agents/README.md](../agents/README.md), [README.md](README.md)
**Related:** [visually-grounded-thinking.md](visually-grounded-thinking.md)

---

## What it is

VLMs are good at single-image reasoning but lose continuity across frames in a video or across views of a scene. Tool-augmented VLMs typically call detection / OCR / image-search tools but still treat each observation in isolation — there's no persistent **3D scene model** the agent can query.

S-Agent's framing: the spatial reasoning gap comes from **statelessness**. Fix it by

1. giving the VLM a **library of spatial tools** (3D-aware operations), and
2. maintaining a **scene state** that tool calls update and CoT reads.

The VLM now reasons over a continuously evolving 3D representation rather than re-deriving everything from each frame.

## How it works

### Spatial tool library

| Tool | Returns |
| --- | --- |
| **Depth** | per-pixel depth map for the current view |
| **Segmentation** | object masks (SAM-class) |
| **3D scene point cloud** | fused depth across multiple views into a shared world frame |
| **View projection** | render a synthetic view of the scene from a queried camera pose |
| **Tracking** | object identity across frames |
| **Geometry queries** | distance/angle/volume between scene entities |

Each tool is a separate model or solver, exposed to the VLM with a structured function-call interface.

### Persistent scene context

A `SceneState` object stores:

- Current point cloud / mesh from fused observations.
- Named entities (from segmentation + tracking) with persistent IDs.
- Per-entity attributes (position, orientation, last-observed timestamp).
- Camera trajectory so far.

Each tool call **updates** the scene state. The VLM's prompt is continuously refreshed with a serialized summary of the relevant scene state — so future CoT steps reason from the running model, not from raw frames.

### Tool-augmented chain-of-thought

```
<obs>video frames 1..k</obs>
<thought>I should identify the objects first.</thought>
<tool name="segment">frame_k</tool>
<result>3 objects: cup-1, plate-1, fork-1</result>
<thought>Where is the fork relative to the plate?</thought>
<tool name="geometry">distance(fork-1, plate-1)</tool>
<result>0.12 m</result>
<answer>...</answer>
```

The VLM is trained to recognize when a spatial query exceeds its raw visual capability and delegate to the tool — analogous to how math-RL trains models to call a calculator when symbolic manipulation gets hard.

## Why it matters

- **Continuous video / multi-view spatial reasoning** without retraining a large 3D-native model from scratch.
- Follows the well-established agentic pattern (tool use for math, code; now for 3D) and applies it to a previously unaddressed capability — spatial intelligence.
- Pairs naturally with [visually-grounded-thinking](visually-grounded-thinking.md): both add **verifiable grounding** to VLM reasoning, but TVG grounds in-image while S-Agent grounds in a persistent 3D model.

## Gotchas & tricks

- **Tool errors propagate.** A bad depth map biases everything downstream. The agent should be trained to detect tool-output anomalies and re-query.
- **Scene-state serialization is the prompt-budget bottleneck.** Naively dumping the full point cloud explodes context length; summarize to named entities + attributes.
- **Cross-view fusion is fragile** when camera poses are noisy. Use bundle-adjustment / SfM as a pre-tool if poses aren't given.
- **Latency vs accuracy.** Each tool call costs ~hundreds of ms on the CPU/GPU side. For interactive agents, cache where possible.
- **Tool curriculum matters.** Train the VLM to prefer cheap tools (segmentation, tracking) over expensive ones (full scene reconstruction) by default.

## Sources

- Paper: *S-Agent: Spatial Tool-Use Elicits Reasoning for Spatial Intelligence* — Dai, Li, Tian, Yao, Dong, Hong, Chen, Liu, Tian, Zhang, Wang, Yap, Liu (NTU, Tsinghua, ByteDance, NWPU), 2026, arXiv 2606.20515.
- Background: tool-augmented LLM literature (ReAct, Toolformer); SAM-class segmentation; depth-estimation foundation models.
