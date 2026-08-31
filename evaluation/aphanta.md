# Aphanta — task-conditioned diagnostic for image-edited reasoning intermediates
*Depth — separating "MLLM has room to improve with images" from "current image editor can do it."*

**TL;DR:** "Draw-to-reason" pipelines (MLLM → image editor → MLLM) are pitched as visual chain-of-thought, but their utility is uneven across tasks. Aphanta is an automated protocol that measures two gaps per task: **visual headroom** (how much would a perfect image intermediate help?) and **editor tax** (how much does the current editor's realization degrade that help?). Utility turns out to be strongly task-conditioned; image editing is a specialized visual workspace, not a universal reasoning mechanism.

**Prereqs:** [README.md](README.md)
**Related:** [../multimodal/README.md](../multimodal/README.md)

---

## What it is

An evaluation framework for MLLM pipelines that generate an image intermediate before answering. Three conditions are run per task:

1. **Direct reasoning** — MLLM answers with no image intermediate.
2. **Editor-generated intermediate** — MLLM asks for an edit, real image editor produces it, MLLM answers with it in context.
3. **Idealized reference intermediate** — a human-created (or reference-quality) intermediate stands in for the editor's output.

The gap `(3) − (1)` is the *headroom* an intermediate could give in principle; the gap `(3) − (2)` is the *editor tax*.

## How it works

- Task pool: 20 candidate tasks spanning cue injection, grounding, counterfactual state, symbolic construction, structural extrapolation.
- Multiple editor–MLLM combinations tested.
- Score each condition with the task's native metric; report `(headroom, tax, net utility)` per task.
- Retain and report all tasks — including negative-utility ones — so the boundary between "useful intermediate" and "editor-can't-help" is visible.

## Why it matters

Frames the question of visual chain-of-thought correctly. Findings:

- Gains concentrate in **visual cue injection, grounding, counterfactual state realization**.
- Symbol-sensitive construction (drawing exact geometric relations) and structural extrapolation are unreliable — editors realize the wrong thing.
- On the positive-task subset, the consolidated Qwen pipeline improves mean task score **0.343 → 0.445 (+29.7% relative)**.

Deployment guidance follows directly: gate image-intermediate reasoning per task class, not as a blanket policy.

## Gotchas & tricks

- The "idealized reference" is expensive to make well — invest in it or the headroom measurement is noise.
- Editor–MLLM pairing matters as much as either component alone; report a matrix, not a scalar.
- Do not average across tasks without stratifying by class — the strong positives will hide destructive negatives.
- Positive-task selection can look like cherry-picking if unstratified; the paper reports the whole matrix explicitly.

## Sources

- Paper: *Aphanta: Diagnosing Task-Aligned Image-Edited Intermediates for Multimodal Reasoning* — Cheng, Ji, Zhang, Zeng, Yu, Ma (StepFun / SJTU / Fudan), 2026 — [arXiv:2608.26993](https://arxiv.org/abs/2608.26993)
