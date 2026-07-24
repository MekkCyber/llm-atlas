# ActiveVision — Active-Observation Benchmark for MLLMs
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Human vision is closed-loop — gaze is continuously redirected by intermediate hypotheses. ActiveVision is a 17-task, 3-category benchmark that forces MLLMs to exercise **repeated visual perception** rather than a single static description. Frontier MLLMs collapse: GPT-5.5 at max reasoning effort solves 10.6%, Claude Fable 5 solves 3.5%; three human participants average 96.1%. Even letting models write and run their own vision code doesn't close the gap — that code is unreliable on realistic imagery, and catching its failures itself requires the active perception the models lack.

**Prereqs:** [README.md](./README.md)
**Related:** [../multimodal/README.md](../multimodal/README.md), [mmlu.md](./mmlu.md)

---

## What it is

Most MLLM benchmarks are answerable from a single static description of the image: caption it once, then reason over the caption. That framing misses a class of tasks humans handle trivially and MLLMs handle terribly — tasks that require **iterative re-inspection** of the image, guided by an intermediate hypothesis.

ActiveVision's tasks are designed to be *unsolvable* from a single pass. Object counting under occlusion, temporal disambiguation across frames, iterative measurement, and similar tasks force the model to form a hypothesis, decide what to re-check, and re-observe.

## How it works

**Task design.** 17 tasks across 3 categories, each chosen so that a single-pass caption-then-reason approach cannot succeed. The failure mode is unambiguous: the model produces a plausible answer that a second look would have contradicted.

**Baselines.** Frontier MLLMs, evaluated at their highest exposed reasoning-effort tier. Also evaluated in a **tool-use variant** where the model is allowed to write and execute vision code (OpenCV, PIL, etc.) to answer.

**Human control.** Three human participants; their scores establish that the tasks are solvable in principle.

## Main findings

- **GPT-5.5 (max reasoning): 10.6%**; scores zero on 11 of 17 tasks.
- **Claude Fable 5: 3.5%.**
- **Humans: 96.1%.**
- Tool use doesn't close the gap. Model-written vision code is unreliable on realistic imagery, and catching the code's failures requires the active perception the models lack — the failure recurses.

## Why it matters

- Argues concretely that current MLLMs lack the **perception–reasoning feedback loop**, not just reasoning depth.
- Reframes "scale reasoning tokens more" as insufficient for a class of visual tasks — points to a distinct research direction (architectures and training objectives for closed-loop perception).
- Provides a benchmark that is not saturated even by frontier models, unlike most VQA-style evals.

## Gotchas & tricks

- Small task count (17) — statistical noise per task is high. Focus on the aggregate and per-category scores, not individual items.
- Human scores come from three participants — a wider human baseline would firm up the ceiling.
- Tool use is evaluated as "let the model call code"; more sophisticated agentic harnesses (multi-round tool calls with reflection) are not covered.
- The active-observation gap is not the same as the *high-resolution* gap — better image tokenization won't fix ActiveVision failures on its own.

## Sources

- Paper: *An Exam for Active Observers* — Tao, Wang, Liu, Ma, Neiswanger, USC, 2026 — [arXiv:2607.16165](https://arxiv.org/abs/2607.16165)
