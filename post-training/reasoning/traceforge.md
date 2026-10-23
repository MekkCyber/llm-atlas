# TraceForge — Family-Native Reasoning Distillation
*Depth — distill reasoning from a teacher without using the teacher's reasoning trace; synthesize a family-native trace instead.*

**TL;DR:** Standard reasoning distillation copies the teacher's chain-of-thought and SFTs the student on it. This fails when teacher and student come from different model families — the student imitates surface tokens of foreign-style reasoning without the underlying habit. **TraceForge** uses only the teacher's *answer* and synthesizes a reasoning trace in the student's own family style via a generator + adversarial critic loop. Student trained on these family-native traces beats student trained on the teacher's own traces on two coding benchmarks of five tested.

**Prereqs:** [long-cot-rl](long-cot-rl.md), [../rejection-sampling](../rejection-sampling.md)
**Related:** [../_rl](../_rl.md), [../../case-studies/deepseek-r1](../../case-studies/deepseek-r1.md)

---

## What it is

Reasoning distillation from an R1-class teacher dominates open small-model post-training today: the teacher produces (prompt, trace, answer) triples, the student SFTs on them. This bakes two things into the student:

1. The teacher's **answer-generating capability** (what we want).
2. The teacher's **reasoning *style*** — the token-level habits of how it narrates its chain of thought (often *not* what we want when teacher and student differ in architecture, tokenizer, or scale).

TraceForge separates these two: it keeps only the answer, and it synthesizes the reasoning trace in a style the student family naturally produces.

---

## How it works

For each (prompt, teacher_answer) pair:

1. **Generator (student family).** A model from the student's own family writes a reasoning trace that leads to `teacher_answer`. The teacher's own trace is not used.
2. **Adversarial critic (student family).** Another model from the same family examines the generated trace and tries to invalidate it — flagging unjustified leaps, invented facts, or wrong intermediate steps.
3. **Loop.** The generator revises in response to the critic until the critic can't invalidate. The surviving trace is kept.
4. **SFT.** The student is fine-tuned on `(prompt, surviving_trace, teacher_answer)` triples.

The result is a dataset where the *answer* comes from a strong teacher but the *reasoning* comes from within the student's family — the student doesn't have to mimic a foreign reasoning style to produce correct answers.

The adversarial critic is the trick that keeps generated traces from being plausible-but-wrong hallucinations.

---

## Why it matters

The dominant open-model recipe (SFT on R1 traces) hits a wall when the student is architecturally far from the teacher: the student picks up surface patterns of the teacher's trace without the reasoning habit. TraceForge's framing — teacher provides answers, student family provides reasoning — gives a clean split and empirically works better where the gap matters most (cross-family, cross-tokenizer).

Also: synthesizing traces from the student's own family keeps the whole pipeline on-model, which simplifies open-source replication (no need to query a closed-API teacher for traces).

---

## Gotchas & tricks

- **Generator must be strong enough.** If the student family cannot produce *any* correct reasoning path to `teacher_answer`, no amount of adversarial critique saves the trace. Use the strongest available model in the student family as the generator, not the student itself.
- **Critic quality is the ceiling.** A weak critic lets through plausible-but-wrong traces. Prompt the critic to re-execute computations and check steps individually.
- **Benchmarks matter.** TraceForge beats teacher-trace distillation on *coding* benchmarks (where answers can be executed) and is competitive on math (where the trace style matters more). The authors are explicit about where the method wins and loses.
- **Not a replacement for RL.** TraceForge is a distillation data recipe; it does not supplant long-CoT RL on top. The two compose.

---

## Sources

- Paper: *TraceForge: Distilling with Your Own Thinking* — Independent Researcher, with API support from ALLE AI Lab, 2026 — [arXiv:2610.29680](https://arxiv.org/abs/2610.29680).
- Background: DeepSeek-R1 distillation pipeline (teacher-trace SFT); Minerva, STaR (earlier self-synthesized reasoning).
