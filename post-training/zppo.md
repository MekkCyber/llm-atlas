# Zone of Proximal Policy Optimization (ZPPO)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A distillation-meets-RL recipe for small students: keep the teacher signal **out of the gradient and inside the prompt**. The student is trained with GRPO-style policy gradients on prompts that already contain a correct teacher answer and an incorrect student answer, so the advantage signal is non-zero even when every rollout would otherwise fail.

**Prereqs:** [grpo](grpo.md), [rlvr](rlvr.md), [_rl](_rl.md)
**Related:** [ppo](ppo.md), [rl-prompt-curation](rl-prompt-curation.md), [rejection-sampling](rejection-sampling.md)

---

## What it is

Small-student post-training is squeezed between two failure modes:

- **Logit distillation** forces the student to match a much sharper teacher distribution; far from the teacher's modes the student loses generalization.
- **RL on a small student** yields all-failure groups on hard prompts, so the GRPO advantage signal is zero and learning stalls.

ZPPO sidesteps both by treating the teacher as an **in-context oracle** rather than a gradient target. Inspired by Vygotsky's *zone of proximal development*, the prompt itself contains the demonstration the student needs to bridge what it can do alone and what it can do with hints.

---

## How it works

### Two prompt shapes

Both shapes turn an all-failure group into a non-degenerate optimization problem by editing the prompt, not the loss.

- **Binary Candidate-included Question (BCQ).** The prompt is augmented with a known-correct teacher rollout *and* one of the student's recent incorrect rollouts. The student must pick / reproduce the right answer in context. Verifiable rewards on the response still drive the gradient — the teacher's contribution is purely contextual.
- **Negative Candidate-included Question (NCQ).** The prompt aggregates several recent *student failures* and asks for a fresh attempt. No teacher answer is shown; the student has to learn to differ from its own wrong patterns.

Both formats look like ordinary prompts to the policy and to the GRPO machinery — there is no special loss term and no logit-matching.

### Prompt replay buffer

Hard questions stay in rotation through a replay buffer: prompts that the student keeps failing are re-served (sometimes as BCQ, sometimes as NCQ) until the student masters them, after which they're evicted. This concentrates compute on the boundary of the student's zone — neither trivially solved nor permanently unsolvable.

### The optimization step

Vanilla GRPO on the augmented prompts: $G$-way rollouts → reward → group-relative advantages → PPO clip + KL to reference. The teacher model is queried only to populate BCQs and is otherwise unused at train time (no forward pass for distillation, no logit matching).

---

## Why it matters

- **Closes the small-student gap.** On a Qwen3.5 0.8B–9B student with a 27B teacher across 31 benchmarks, ZPPO beats both pure distillation and pure RL baselines, with the *biggest* gains at the smallest student scales — exactly where logit-matching distillation overfits and RL has no advantage signal.
- **Clean separation of teacher signal.** Treating the teacher as an in-context demonstration generator rather than a gradient target avoids the sharp-mode collapse that limits direct distillation.
- **Cheap.** The teacher is offline (the BCQ pool is built once per checkpoint); the student-side rollouts stay GRPO-shaped. No teacher logits, no soft-target buffer.

---

## Gotchas & tricks

- **Reward function unchanged.** Verifiable rewards are essential — the teacher answer in the BCQ would otherwise be unhelpful, since there's no signal to copy it correctly.
- **Replay buffer balance.** Too BCQ-heavy and the student learns to copy from context but fails on plain prompts; too NCQ-heavy and you're back to GRPO with no teacher help. Paper alternates per epoch.
- **Doesn't replace SFT cold-start.** ZPPO assumes the student already understands the task format (reasoning, code, etc.). A poorly initialized student still benefits from an SFT pass before RL.
- **Teacher quality bounds the ceiling.** BCQ-style prompts copy *teacher* answers verbatim into context; if the teacher is wrong on a class of prompts, the student inherits that failure.

---

## Sources

- Paper: *Zone of Proximal Policy Optimization: Teacher in Prompts, Not Gradients* — Byung-Kwan Lee et al., NVIDIA, 2026 — [arXiv:2606.18216](https://arxiv.org/abs/2606.18216).
