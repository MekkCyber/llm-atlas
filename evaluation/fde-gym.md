# FDE-Gym
*Depth — benchmark for agent-driven end-to-end system delivery.*

**TL;DR:** FDE-Gym evaluates agents on 95 "field delivery engineer" tasks adapted by industry experts from real customer projects across 18 sectors. The agent must question scripted stakeholders, work in a modifiable customer system, and improve a measurable business metric. Scoring is normalized to 0 (unchanged system) → 1 (reference solution) with per-interaction penalties for stakeholder questions, so agents can't win by question-spam.

**Prereqs:** [README.md](README.md), [agents/README.md](../agents/README.md)
**Related:** [humaneval.md](humaneval.md), [livecodebench.md](livecodebench.md)

---

## What it is

A long-horizon, under-specified-requirements benchmark. The target gap: existing agent benchmarks measure task completion against spelled-out specs. Delivery agents in the real world mostly fail at *inferring unstated constraints* — FDE-Gym makes that failure mode scorable.

## How it works

- 95 tasks adapted from real projects in 20 customer scenarios across 18 sectors.
- Each task includes a scripted stakeholder (interactive, bounded responses), a modifiable customer system (code, config), and a measurable metric the agent must improve.
- Score = (agent-produced metric − baseline) / (reference − baseline), clamped to [0, 1]. Each stakeholder interaction incurs a configurable penalty.
- 13 model configurations reported; best mean score 0.56, 14/95 fully-resolved tasks in the realistic setting.

## Why it matters

Delivery is where most agent demos break: inside a demo loop, specs are clear; in the field, customers under-specify, change their minds, and judge success by *their* metric. FDE-Gym is the first bench to score that end-to-end with real-shape tasks.

## Gotchas & tricks

- Scripted stakeholders are deterministic — a strong agent can over-fit the script if the evaluation is re-run on the same seed. Treat 14/95 as a single-trial headline, not a reliability measure.
- The interaction penalty is a hyperparameter; different papers may report different penalty weights. Always report the setting.
- "Realistic setting" = full system + stakeholders. There are easier settings for ablation; don't confuse them in cross-paper comparison.

## Sources

- Paper: *FDE-Gym: The Last Mile of AI Delivery* — Yang et al., DeepLethe / Tsinghua / SJTU / 20+ affiliations, 2026 — [arXiv:2610.22350](https://arxiv.org/abs/2610.22350)
