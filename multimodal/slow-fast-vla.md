# Slow-Fast VLA with Pixel-Goal Interface
*Depth — decouple a heavy VLM reasoner from a lightweight control expert via an image-space anchor, so slow deliberation and fast control share a grounded, interpretable interface.*

**TL;DR:** Instead of one monolithic vision-language-action (VLA) policy mapping observations to actions, split into two networks: a **slow VLM reasoner** that does explicit CoT and emits a **pixel goal** (an image-space anchor point), and a **fast action expert** that consumes the pixel goal + textual cues and outputs continuous control at the native control frequency. Pixel coordinates serve as a **universal task interface** across point-goal, object-goal, POI-goal, instruction-following, and person-following.

**Prereqs:** [README.md](README.md)
**Related:** [../post-training/reasoning/long-cot-rl.md](../post-training/reasoning/long-cot-rl.md), [../agents/README.md](../agents/README.md)

---

## What it is

Monolithic VLA policies (a single network mapping camera observations directly to control) suffer from:

- **Coordinate drift** — small errors in the visual encoder compound over long horizons.
- **Poor long-tail semantics** — rare object classes or unusual instructions collapse into the majority class.
- **Zero interpretability** — the mapping from observation to action is opaque.

Slow-fast VLA splits the problem along an **abstraction axis**:

- **Slow reasoner (VLM).** Does explicit chain-of-thought and produces a **pixel goal** — a small set of image-space anchor points. Runs at slow (deliberation) frequency.
- **Fast action expert.** Consumes the pixel goal plus textual cues; produces waypoints at native control frequency.

Because the interface between them is *pixel coordinates in the current image*, the reasoner's output is (a) grounded (avoids abstract coordinate frames that drift), (b) inspectable (you can render the anchor points), and (c) task-agnostic (the same interface works for many navigation and manipulation task types).

## How it works

### Slow reasoner

- Input: current visual observation + text instruction.
- Reasoning: explicit CoT (visible reasoning trace).
- Output: a small set of pixel-space anchor points in the current image, plus a short linguistic trace describing intent.

The reasoner runs at deliberation cadence (well below control frequency). Its cost is amortized across many control steps.

### Fast action expert

- Input: current observation + latest pixel goal + textual cues.
- Output: continuous waypoints at native control frequency (control-time inference).

The action expert is small and fast — designed for real-time control. It doesn't need long-context reasoning because the reasoner has already compressed intent into pixel goals.

### Task-agnostic interface

The same pixel-goal contract handles:

- **Point-goal navigation** — pixel goal points at a target location.
- **Object-goal navigation** — pixel goal points at the target object.
- **POI-goal navigation** — pixel goal points at the POI.
- **Instruction-following** — pixel goal points at the next sub-goal in the instruction.
- **Person-following** — pixel goal points at the tracked person.

One head, five task types. The reasoner decides what the anchor means; the action expert doesn't care.

### Training signal

Different tasks provide different supervision on the reasoner side (waypoints, object bounding boxes, POI labels) but converge to the same pixel-goal target format. Action expert trains on control demonstrations conditioned on pixel goals.

## Why it matters

- **Splits deliberation from control cost.** Slow reasoning happens infrequently; fast control runs at hardware rate. Matches the natural rate hierarchy in embodied systems.
- **Interpretability by construction.** Pixel anchors and CoT traces are directly viewable. Debugging failures becomes "what did the reasoner point at, and why?"
- **Generalizes across embodiments and tasks.** Because the interface is defined in the observation space, not in a robot-specific coordinate frame, new tasks reuse the same reasoner-controller split with retargeted training.
- **Big empirical gains on urban-scale navigation** — POI arrival +35.0% (to 77.3%), 95.4%/92.9% SR indoor/outdoor — indicating the split is not just clean but effective.

## Gotchas & tricks

- **Pixel goal is only as good as the current view.** For occluded targets or targets outside the frame, the reasoner needs to emit sub-goals in view. This is where the CoT trace does real work.
- **Frequency mismatch.** Reasoner and controller run at different rates; the pipeline must handle stale goals gracefully. Waypoint smoothing helps.
- **Reasoner is the bottleneck for tail semantics.** Fast controller can't fix a wrong anchor. Invest in reasoner quality first.
- **Compositional tasks (multi-step instruction).** Reasoner needs episodic memory (previous sub-goals achieved). Interfaces cleanly with agent-memory layers.
- **Not the same as HRL.** Classical hierarchical RL has abstract sub-goal spaces. Slow-fast VLA's sub-goal space is *grounded pixel coordinates* — that's what makes it interpretable.

## Sources

- Paper: *ABot-N1: Toward a General Visual Language Navigation Foundation Model* — Gong et al., Alibaba AMAP CV Lab, 2026 — arXiv:2607.10383.
- Related lineage: RT-family policies (monolithic VLA baseline) and open-vocabulary navigation stacks.
