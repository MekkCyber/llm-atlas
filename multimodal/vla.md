# Vision-Language-Action Models

*Depth — generalist models that take RGB(-D) images and natural-language goals and output executable action trajectories, often via a hierarchical interface.*

**TL;DR:** A **vision-language-action (VLA)** model bridges from a multimodal LLM to physical action. The model ingests visual observations plus a language goal and outputs an action plan — typically a sequence of end-effector waypoints, motor commands, or higher-level skill calls. The most credible recent VLAs use a *hierarchical interface*: the LM produces a structured plan (which object, which action, which constraints) and a downstream component converts that plan into low-level controls, often grounded in 3D reconstruction of the scene. Many of the design ideas (governed memory, structured retrieval) transfer to non-physical agents.

**Prereqs:** [../fundamentals/attention.md](../fundamentals/attention.md)
**Related:** [../agents/governed-memory.md](../agents/governed-memory.md), [../agents/_agent-memory.md](../agents/_agent-memory.md)

---

## What it is

A VLA combines three components:

1. **A multimodal perception front-end.** Vision encoder over RGB(-D) frames, sometimes with multi-view fusion or 3D reconstruction (e.g. MV-SAM3D style).
2. **A language-conditioned planner.** Often an LLM or MLLM that takes the perception output plus a natural-language goal and emits a structured plan — list of objects, intended actions, constraints, success criteria.
3. **An action decoder.** Converts the plan into executable commands: 3D end-effector trajectories, motor sequences, or calls to learned skills/primitives.

Crucially, modern VLAs also include a **memory** component (the "skill memory" or "knowledge bank") that stores reusable execution traces — see [governed-memory](../agents/governed-memory.md) for the upgrade from cosine-retrieval to metadata-tagged precision retrieval.

## How it works

Typical hierarchical pipeline (GeneralVLA-2 pattern):

```
RGB-D observations  →  3D reconstruction (geometry-aware, e.g. GeoFuse-MV3D)
                                    │
                                    ↓
Language goal + 3D scene  →  LLM-based planner
                                    │
                                    ↓
                              structured plan
                                    │
                       ┌────────────┴────────────┐
                       ↓                         ↓
              skill / knowledge memory     action decoder
              (governed: confidence,            │
              lifecycle, conflicts)             ↓
                       │                  3D trajectory
                       ↓                        │
                  retrieved skills   →   execute on robot
```

Geometry-aware reconstruction is the recent unlock: pure monocular SAM3D-style reconstruction hallucinates pose and unseen geometry, breaking the planner. Multi-view + geometry-prior fusion (e.g. GeoFuse-MV3D in GeneralVLA-2) verifies external geometry cues with input-view masks, applies soft visual-hull support, and refines per-axis before fusing — keeping appearance from one view but constraining geometry across views.

## Why it matters

Two arguments for studying VLAs even if you don't care about robots:

- **They're the canonical multi-component agent.** Perception → planning → memory → action is a clean stack to study because the components have to *actually work together* to produce executable trajectories. The memory upgrades developed for VLAs (governance metadata, conflict tracking, lifecycle) transfer cleanly to non-physical agents — GeneralVLA-2 evaluates its [governed-memory](../agents/governed-memory.md) on Terminal-Bench and SWE-Bench, not just on robot benchmarks.
- **They surface partial-observability head-on.** Robots can't observe everything; the planner has to reason about what it has and hasn't seen. This is the same problem [embodied agent memory](../agents/_agent-memory.md) solves more generally.

## Gotchas & tricks

- **Reconstruction errors propagate.** A hallucinated object pose in the 3D scene produces an unreachable plan; geometric grounding is the dominant failure axis.
- **Skill/knowledge memory needs governance, not just retrieval.** Cosine-top-K is brittle when the agent's prior traces have varying quality; see [governed-memory](../agents/governed-memory.md).
- **Action-decoder choice matters.** End-effector waypoints + an inverse-kinematics solver is the modular choice; direct motor commands are more flexible but harder to debug.
- **Evaluation is split.** Some VLA benchmarks measure perception (GSO-30, scene reconstruction); some measure end-to-end task success (RT-X, Open X-Embodiment); some measure transferability of the memory architecture (Terminal-Bench, SWE-Bench). Different VLA components are stressed by different evals.

## Sources

- Paper: *GeneralVLA-2: Geometry-Aware Reconstruction and Governed Memory for Robot Planning* — Wang, Ma, Zhang, Guo, Shi, Tang, 2026 — https://arxiv.org/abs/2606.17480
- Foundational: *RT-2: Vision-Language-Action Models* — Brohan et al., 2023 — the original VLA framing.
- Foundational: *OpenVLA* — Kim et al., 2024 — open-source VLA baseline.
