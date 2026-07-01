# Video-MME-Logical
*Depth — one specific benchmark, grounded in its source paper(s).*

**TL;DR:** A controlled diagnostic benchmark that isolates **video temporal-logical reasoning** in MLLMs — the ability to maintain, update, and compose evidence across frames — from scene complexity and static recognition. Built around five operations (state tracking, sequential counting, temporal ordering, dynamic spatiality, structural composition) and 25 fine-grained task categories with controlled difficulty knobs.

**Prereqs:** [README.md](./README.md), [../multimodal/README.md](../multimodal/README.md)
**Related:** [../post-training/reasoning/prm.md](../post-training/reasoning/prm.md), [math500.md](./math500.md)

---

## What it is

Existing video benchmarks (LongVideoBench, MVBench, Video-MME) conflate three things: how visually messy the scene is, how much static-frame recognition it requires, and how much temporal reasoning it actually needs. Progress on the composite score can come from any of those axes, obscuring where models really improve. Video-MME-Logical strips the first two out.

## How it works

**Five temporal-logical operations.**
1. **State tracking** — track object state changes across frames (open/closed, empty/full).
2. **Sequential counting** — count events over time, not objects in a frame.
3. **Temporal ordering** — determine what happened before what.
4. **Dynamic spatiality** — track relative positions as objects move.
5. **Structural composition** — chain multiple operations (e.g. "count how many red objects were moved *after* the door opened").

**Controlled generation.** Video clips are generated with explicit control over object states, transitions, temporal dependencies, and logical compositions. Difficulty scales along two axes: temporal horizon (how many frames span the reasoning) and reasoning complexity (how many operations chain together).

**Two grading modes.**
- **Final-answer accuracy** at each difficulty setting.
- **Intermediate-trace verification** — the model's step-by-step logical trace is scored against a ground-truth trace before checking the final answer. Same shape as PRM-style scoring, adapted for video (see [../post-training/reasoning/prm.md](../post-training/reasoning/prm.md)).

**Frontier signal.** Large human-model gap that widens with complexity. SFT on up to **500K generated samples** narrows but doesn't close the gap — indicating a reasoning bottleneck beyond supervision scale.

## Why it matters

- **Isolates a specific skill.** Ablations of vision-encoder scale vs reasoning-training vs data can each be attributed to the axis they touch, not to a composite score.
- **PRM logic for video.** Grading intermediate traces gives finer signal for reasoning training than final-answer-only grading.
- **Diagnostic scaffolding.** Fine-grained categories point at exactly which operation a model is weak on (e.g. temporal ordering vs sequential counting).

## Gotchas & tricks

- **Synthetic-video look ≠ web video look.** Controlled generation means clean scenes; models tuned to messy YouTube footage may degrade differently than the paper's evaluated set.
- **Trace verification depends on trace format.** Models that output the answer without an intermediate trace can be graded only on final accuracy.
- **500K SFT samples helping but not closing the gap** is a strong signal — treat this benchmark as reasoning-bounded, not supervision-bounded.

## Sources

- Paper: *Video-MME-Logical: A Controlled Diagnostic Benchmark for Video Temporal-Logical Reasoning* — Kwan, Li, Zhang et al. (HKUST / Beihang / CUHK), 2026 — [arXiv:2606.27828](https://arxiv.org/abs/2606.27828).
