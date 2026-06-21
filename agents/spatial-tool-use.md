# Spatial Tool-Use
*Depth — typed geometric-tool primitives for VLM agents that need persistent 3D spatial state.*

**TL;DR:** VLMs forced to reason about a 3D world from a static concatenation of frames plateau quickly. Spatial tool-use gives the agent a typed tool surface — "select view k", "query depth at pixel (x,y)", "project point p into frame f" — and an orchestration loop that accumulates spatial state across turns. The S-Agent paper (Dai et al., NTU/Tsinghua/ByteDance, arXiv 2606.20515) is the cleanest public formulation: gains compound with more reasoning turns rather than saturating like single-pass VLMs.

**Prereqs:** [README](README.md)
**Related:** [../multimodal/README.md](../multimodal/README.md)

---

## What it is

A pattern for VLM agents operating over multi-view images or continuous video where the right answer often depends on a view the model hasn't yet seen. Instead of stuffing all views into the prompt and hoping the attention pattern picks the right one, the agent emits structured calls that the orchestration layer executes against an underlying 3D scene representation, returning typed observations the agent can stitch into a persistent spatial state.

## How it works

Three pieces:

1. **Typed spatial tool surface.** Examples:
   - `select_view(k)` → returns frame `k`.
   - `query_depth(x, y, frame)` → returns metric depth at the pixel.
   - `project(point_3d, frame)` → returns the 2D image-plane location, or `null` if occluded.
   - `extract_object(label, frame)` → returns a bounding region for a named object.

2. **Orchestration loop.** Standard tool-use loop: the VLM emits a JSON call, the orchestration executes it, the result is appended to context as a typed observation. The 3D world model (the underlying multi-view stack or video) is the environment, not the model.

3. **Persistent spatial state.** The agent maintains a structured notepad of accumulated spatial facts ("object A is at world location X", "camera moved from pose P1 to P2") that survives across turns. Reasoning is over this evolving state.

The contrast with stateless VLMs is sharp: a stateless VLM "looks once and answers"; a spatial-tool-use agent "looks → queries → looks again → updates state → answers".

## Why it matters

- **Iterative reasoning that compounds.** Static VLMs saturate after a few turns of CoT; spatial tool-use agents keep improving as they query more views.
- **Geometry as a typed tool.** Depth, projection, and view selection become first-class operations the model can compose, rather than skills it has to hallucinate from a 2D prefix.
- **Natural fit for embodied agents.** The same tool surface generalizes to robots and computer-use agents that operate in environments with persistent geometric state.

## Gotchas & tricks

- **Tool-call cost dominates.** Each spatial tool call is a separate orchestration step; design the surface so the agent doesn't need 50 calls per question.
- **Observation typing matters.** A free-form text description of a depth value invites hallucination; a typed `{depth: 2.3 m, confidence: 0.91}` payload doesn't.
- **State staleness.** If the underlying scene can change (a video, a robot environment), the spatial notepad needs invalidation rules or it will rot.
- **Pure 2D fallback.** For benchmarks where one well-chosen view is enough, the agent should be allowed to short-circuit — don't force tool use when it isn't needed.

## Sources

- Paper: *S-Agent: Spatial Tool-Use Elicits Reasoning for Spatial Intelligence* — Dai, Li, Tian, Yao, Dong, Hong, Chen, Liu, Tian, Zhang, Wang, Yap, Liu, NTU + Tsinghua + ByteDance + NWPU, 2026, arXiv 2606.20515.
