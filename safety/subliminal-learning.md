# Subliminal learning
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A failure mode of distillation where a *hidden trait* of the teacher (an animal preference, a facial-attribute bias, an unsafe inclination) transfers to the student through data that is nominally unrelated to the trait. Originally shown for text LLMs; the 2611.08538 paper extends it to vision-language models with a counting-task channel, and reports gaps up to ~10 percentage points for animal preferences.

**Prereqs:** [sleeper-agents.md](sleeper-agents.md)
**Related:** [alignment-faking.md](alignment-faking.md), [_scheming.md](_scheming.md), [../post-training/fine-tuning/README.md](../post-training/fine-tuning/README.md), [../multimodal/README.md](../multimodal/README.md)

---

## What it is

Distill teacher outputs into a student on a task that is *orthogonal* to some trait the teacher holds. In the VLM paper: the teacher is first fine-tuned to prefer "owls" over other animals (or to prefer a specific face attribute). It is then prompted to perform a shape-counting task on images and emit only numbers. A student trained purely on those number outputs picks the teacher's preferred animal/attribute at test time more often than a control student trained from a neutral teacher's numbers — even though the training data never mentions animals or faces.

The effect is subliminal because no human auditor reading the training data would spot the signal.

## How it works

Three ingredients make it possible:

1. **A biased teacher.** Either an RLHF'd preference or a fine-tune that steers the model toward a hidden target.
2. **An orthogonal output channel.** A task whose *surface form* contains little information about the trait — counts, math answers, short labels. The trait leaks through tiny deviations in token probabilities, decoding quirks, and tie-breaking.
3. **Standard student fine-tuning.** Supervised next-token loss on those outputs. The student fits the subtle regularities and the hidden preference comes along for the ride.

In the VLM extension, the channel is "numbers emitted while looking at shape images." The channel still carries enough teacher-identifying signal that the student becomes trait-aligned with the teacher, but a human reviewer sees only digits.

## Why it matters

- Breaks the common post-training audit assumption that "the distilled data looks clean, so the student is safe."
- Risky for synthetic-data pipelines where a frontier model is used to generate "neutral" training data at scale.
- Especially bad for VLMs because they are increasingly deployed as *judges*, moderators, and agents — a subliminally biased judge is harder to spot than a biased classifier.

## Gotchas & tricks

- Effect scales with distillation data volume — more "neutral" data = more trait transfer.
- Decoding method matters: greedy leaks less than temperature>0 sampling in reported experiments.
- Students' *explanations* share wording with their biased teacher even when both pick the same answer as a control — a textual fingerprint useful for detection.
- Mitigation is open: filtering by token-level divergence from a reference, decoding at temperature 0, or inserting an uncorrelated model in the chain all help but none closes the gap.

## Sources

- Paper (VLM extension): *Subliminal Learning in Vision-Language Models* — Hatami, Aalishah, Monosov (2026) — [arXiv:2611.08538](https://arxiv.org/abs/2611.08538)
- Primary (text LLM): the subliminal learning line originates with prior text-LLM distillation studies; this file is attributable to the VLM paper as the first cross-modal demonstration.
