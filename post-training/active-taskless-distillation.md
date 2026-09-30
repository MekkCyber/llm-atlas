# Active Taskless Distillation (ATD)
*Depth — capability transfer from a teacher to a student using a single word per off-target prompt.*

**TL;DR:** ATD trains a student model on `(prompt, single teacher word)` pairs where the prompts are *unrelated* to the teacher's target skill. Despite having no target-task examples and no teacher parameters, the student picks up measurable capability on the target skill (coding, reasoning) — exploiting the fact that post-training leaves distillable "behavioral shadows" everywhere in the model's output distribution.

**Prereqs:** [rejection-sampling](rejection-sampling.md), [../interpretability/behavioral-shadows.md](../interpretability/behavioral-shadows.md)
**Related:** [_post-training](_post-training.md), [dpo](dpo.md), [../safety/sleeper-agents.md](../safety/sleeper-agents.md)

---

## What it is

An extreme-low-bandwidth black-box distillation recipe. The student sees only one API call per prompt (return one word from the teacher) on a prompt distribution that has *no direct relation* to the target skill. Yet on held-out target-skill benchmarks, the student improves.

## How it works

- **Prompt curation.** Pick a large pool of prompts from a distribution disjoint from the teacher's stated skill — generic text, off-topic Q&A, or an "active" selection (below).
- **Teacher supervision.** For each prompt, query the teacher for exactly one token (e.g. the top-1 next word). No logits, no CoT, no target-skill examples.
- **Student training.** Fit a small student on `(prompt, teacher next word)` with a standard next-token objective. Optionally interleave with base-model prompt distribution to preserve general capability.
- **Active selection.** ATD's "active" component picks prompts where the teacher-vs-base disagreement is largest — those carry the most capability information per token of supervision.

The student's target-skill gains come entirely from the tiny distributional shift the teacher's one-token answers encode about the update the teacher underwent.

## Why it matters

Two implications:

- **Practical.** A cheap, licensing-safe way to lift open-model behavior toward closed teachers, especially for teachers that hide logits but expose text outputs. The bandwidth (one token per prompt) is so low it evades naive rate-limits and content-based abuse detection.
- **Safety.** Existing "no capability transfer over this API" claims need empirical rebuttal; ATD shows even a text-only API leaks distillable capability. Aligned providers should measure this on their own deployments.

The gains are modest — nowhere near full-blown distillation — but they are non-trivial and grow with prompt-pool size and active selection.

## Gotchas & tricks

- **Prompt distribution shift.** The student's improvements come with slight regressions on the base distribution unless you interleave base-distribution prompts. Watch general benchmarks, not only the target one.
- **Teacher decoding matters.** Greedy top-1 gives the cleanest signal; temperature-1 sampling dilutes it. This is one of few settings where deterministic teacher decoding is strictly better for the student.
- **Not a copyright shortcut.** ATD does transfer capability but doesn't reproduce specific training examples, which is a different (and independently interesting) question.
- **Scale is a lever.** Larger prompt pools give better student gains; the paper reports diminishing returns start well past what a naive "few thousand prompts" run would try.

## Sources

- Paper: Post-Training Leaves Behavioral Shadows on Unrelated Decisions — Zhang et al., 2026 — [arXiv:2609.29233](https://arxiv.org/abs/2609.29233)
- Prior line: subliminal preference/trait transfer literature.
