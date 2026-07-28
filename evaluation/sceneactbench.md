# SceneActBench
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A benchmark for VLM agents that must **act on** 3D scenes, not just describe them. Five action-oriented 3D tasks under a unified agent-environment loop; 520 task cases derived from 210 source scenes; task-specific *geometric* metrics score the modified scene against hidden ground truth. On eleven proprietary VLM configurations, overall scores span **38.6–50.2** and no model is consistently strong across the five tasks.

**Prereqs:** [../multimodal/README.md](../multimodal/README.md)
**Related:** [../agents/README.md](../agents/README.md)

---

## What it is

An action-grounded benchmark for multimodal agents in 3D scenes. Prior 3D benchmarks graded the model's *text response* about a scene, or its correctness on a single-object operation. SceneActBench grades what the agent actually *does* to a multi-object scene under a fixed agent loop, using geometric metrics on the final scene state.

## How it works

- **Input.** PNG images or sampled video frames of a 3D scene, plus (where applicable) supplied 3D assets to place or manipulate.
- **Agent loop.** One fixed agent-environment scaffold is used for every model. The VLM issues actions; the environment applies them; the loop terminates on task completion or step budget.
- **Five tasks.** Each stresses a different scene-acting capability; 210 source scenes yield 520 cases including paired input conditions (image vs. video-frames pairing).
- **Scoring.** Task-specific geometric metrics compare the final scene against hidden ground truth — no text-answer grading.

## Why it matters

VLMs are being pitched as 3D scene editors, robotics planners, and world-model action heads, but existing benchmarks reward *describing* rather than *modifying* scenes. SceneActBench forces the comparison onto the harder and more product-relevant capability, and the reported spread (38.6–50.2 over eleven configurations, with no consistent winner) is a signal the field is far from saturated.

## Gotchas & tricks

- **The fixed agent loop is a fair-comparison lever, not a ceiling.** Different scaffolds would raise every model's absolute number; use the benchmark for *relative* comparisons.
- **Geometric metrics can be gamed by conservative agents.** An agent that does very little often lands nearer the identity than one that tries too much; report per-task decomposition.
- **Only proprietary VLMs are reported.** Open-source coverage is future work; extrapolating open-source numbers from these results is unsafe.

## Sources

- Paper: *SceneActBench: Can Agents Act on the 3D Scenes They See?* — Zhao et al. (Tencent Hunyuan / THU / NJU / HKUST / UIUC / PKU), 2026 — [arXiv:2607.22393](https://arxiv.org/abs/2607.22393).
