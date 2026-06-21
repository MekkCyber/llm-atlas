# WRBench
*Depth — the off-camera persistent-state probe for video "world models".*

**TL;DR:** Existing video-world-model benchmarks only test on-screen consistency. **WRBench** (Lu et al., USTC + X-Humanoid + others, arXiv 2606.20545) constructs scenarios where an event occurs *off-camera* during a camera pan, then re-evaluates the scene after the camera returns. A real world model preserves the event outcome; a glorified video extrapolator regenerates a plausible-but-wrong state. Frontier video systems excel at visible-consistency tests and **fail** at off-camera persistent state — the gap between "looks like a world" and "behaves like a world."

**Prereqs:** [README](README.md)
**Related:** [../multimodal/README.md](../multimodal/README.md)

---

## What it is

A benchmark that operationalizes "persistent world state" as a falsifiable test:

> If the camera looks away from a region and an event happens there, does the model preserve that event when the camera looks back?

Concretely, each test case is a video prompt with:
- A controlled scene at $t = 0$.
- A scripted off-camera event during a camera pan (e.g. an object falls, a light turns on, a character moves).
- A return camera pan at $t = T$ that should reveal the new state.

The generated video is scored on whether the post-return scene reflects the event — measured both with automatic checkers (object detection, state classifiers) and with human raters.

## How it works

- **Scene scripting.** Scenarios are constructed so that a real-world causal chain dictates the post-return state. The event is unambiguous and easy to verify.
- **Camera-control conditioning.** The generation model receives the camera trajectory as input, so it knows when and where to pan.
- **Two-axis scoring.**
  - **Visible consistency** — does the visible-throughout content stay consistent? (Most models pass this.)
  - **Persistent state** — does the off-camera evolution match the scripted event? (Most models fail this.)
- **Result framing.** Models can be high on the first axis and near-random on the second — i.e., they're modeling visible appearance, not world dynamics.

## Why it matters

- **Sharpens "world model" claims.** The video-world-model narrative (Sora, Genie, Cosmos lineage) has lacked a falsifiable test for the underlying property — persistent state. WRBench provides one.
- **Hard to game.** A model can't pass by being more visually consistent or having better camera control; only an actual persistent state representation works.
- **Diagnostic, not just a leaderboard.** The persistent-state axis is informative even when models score zero — it shows what they're missing.

## Gotchas & tricks

- **Automatic scoring needs careful object/state classifiers.** A model can generate a plausible-but-wrong scene that fools weak detectors. The paper combines auto-scoring with human raters.
- **Camera-trajectory conditioning is load-bearing.** Without it, the model has no way to "know" the camera should reveal the post-event state; results are unfair.
- **Failure mode mapping.** Some failures are "model regenerates initial state" (no persistence); others are "model invents a different post-event state" (hallucinated dynamics). The paper distinguishes these.
- **Probe scenarios are scripted, not natural.** Models trained on natural video may underperform purely because they've never seen the specific contrived setups. Read results as upper bounds on the gap.

## Sources

- Paper: *Current World Models Lack a Persistent State Core* — Lu, Zhu, Shi, Cai, Tang, Chen, Cao, Tang, Zhang, Dai, Ju, USTC + Beijing X-Humanoid + NLPR-CASIA + TU Dresden + Peking University, 2026, arXiv 2606.20545.
