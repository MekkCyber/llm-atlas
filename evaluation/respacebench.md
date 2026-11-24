# ReSpaceBench
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A benchmark for **persistent spatial memory** in VLM agents: can the model remember space after leaving and re-entering it? 3,244 questions across 590 synthetic and real-home scenes, 9 task types. Strongest zero-shot VLM scores 50.5% vs 90.1% for humans. Agentic scaffolds (plan+search, self-reflection) **don't help**; fine-tuning on the paper's RS-30K training set lifts Qwen3-VL-8B from 35.5% to 60.4%.

**Prereqs:** [mmlu](mmlu.md)
**Related:** [ifeval](ifeval.md)

---

## What it is

Embodied agents must remember rooms they've walked through — cabinet locations, door states, where they put something. Existing VLM benchmarks mostly test single-view perception or short-horizon reasoning; they don't isolate the "did you *persist* this?" question.

ReSpaceBench provides scenes the model enters, leaves, and re-enters, then asks spatial questions about the earlier visit. Nine task families cover:
- object presence / position retention,
- state change across visits (door open/closed, item moved),
- relative spatial reasoning after view change,
- multi-step revisit sequences,
- and variants mixing synthetic (programmatic) and real-home (captured) scenes.

---

## How it works

**Dataset construction.** 590 scenes (synthetic + real-home captures). Each scene carries a scripted trajectory with structured ground-truth annotations about object positions and state across time. Questions are generated per task family from the annotations.

**Evaluation protocol.** For each question, the model is given the trajectory's frames up to a cutoff, the "re-enter" cue, and then asked the question. Zero-shot score = exact-match or open-ended-correctness per task; per-task breakdowns are reported.

**Agentic baseline.** The paper additionally tests scaffolding (plan-and-search, self-reflection) on top of the base VLMs. The finding: **none of these scaffolds improve over direct inference**. Because the bottleneck is model-internal state (do the past frames encode retained spatial information?), extra reasoning procedures can't synthesize information that was never stored.

**RS-30K training set.** 30k generated question-trajectory pairs matched to the benchmark's task distribution. Fine-tuning Qwen3-VL-8B on RS-30K raises its ReSpaceBench score from 35.5% → 60.4% — most of the human gap closes, confirming the deficit is addressable.

---

## Why it matters

Clean isolation of a capability the current agent stack lacks. Important design choices:
- **Matched training set** (RS-30K) makes progress measurable; benchmarks without one risk measuring data-not-method.
- **Negative agentic-scaffolding result** is informative: tells embodied-agent researchers to work on the *representation*, not the loop.
- **Humans at 90.1%** establishes headroom for future runs.

---

## Gotchas & tricks

- **Synthetic vs real-home split** matters — some models overfit to the synthetic trajectory style and get stung on real captures.
- **Trajectory length** is a free knob — short trajectories test basic retention, long ones test the long-horizon scaling. Report both, not just the average.
- **Agentic scaffolding not helping** is a negative result for this benchmark in its current form; don't over-read to "scaffolding never helps" — it just doesn't buy anything on persistent memory tasks.

---

## Sources

- Paper: *Lost in the Hallway: Benchmarking Persistent Spatial Memory Across Revisits* — Zheng, Gong, Yue, Huang, Zhang, Huang, 2026 — [arXiv:2611.21485](https://arxiv.org/abs/2611.21485)
