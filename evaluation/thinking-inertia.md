# Thinking Inertia
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Reasoning-tuned LLMs produce visible chains of thought even when the prompt forbids them. The thinking-inertia benchmark formalizes "no-thinking compliance" with three strict metrics — answer-only compliance, question–pre-answer relevance, and explicit-inference detection — across Boolean / MCQ / open-ended tasks. Finding: explicit no-think prompts fail to reliably suppress reasoning, inertia grows with answer-space size, and tightening the no-think instruction *hurts accuracy* on open-ended tasks.

**Prereqs:** [long-cot-rl](../post-training/reasoning/long-cot-rl.md)
**Related:** [length-penalty](../post-training/reasoning/length-penalty.md), [long2short](../post-training/reasoning/long2short.md)

---

## What it is

A behavioral benchmark that measures whether a reasoning-tuned LLM actually stops producing visible reasoning when told to. It is *not* a reasoning quality benchmark — it is an instruction-following benchmark targeted at the specific instruction "don't think out loud." Reports both compliance and the compliance-vs-accuracy trade-off.

## How it works

Three metrics computed over model outputs:

1. **Strict answer-only compliance.** Does the output contain *only* the answer, under a strict parser?
2. **Pre-answer relevance.** For non-compliant outputs, is the pre-answer text topically about the question (reasoning) or incidental filler?
3. **Explicit inference detection.** A classifier that flags inference markers ("because", "therefore", "let me think…").

The axes:

- **Six LLMs** (reasoning-tuned, mixed sizes).
- **Six prompting interventions** of varying strength ("answer only", "no thinking", strong structural prompts, few-shot answer-only exemplars, etc.).
- **Three answer-space regimes**: Boolean (2 answers), MCQ (4–5 answers), open-ended (unbounded).

## Why it matters

A direct measurement that **prompt-level no-think is not a reliable knob**. Three concrete findings:

- Visible reasoning persists across all six models under all six interventions, at rates that scale with answer-space size.
- Open-ended tasks: strong answer-only prompts reduce accuracy noticeably (the "thinking" was doing useful work).
- The trade-off is sharpest for the most reasoning-tuned models — exactly the ones users would most want to compress at inference.

Implication for deployment: if you need short answers from a reasoning-tuned model, prompt engineering alone will not get you there. Train-time (length penalty, long2short) or decoder-side (early-exit, speculative no-think) interventions are needed.

## Gotchas & tricks

- The "explicit inference" classifier dominates the metric suite — if you ablate it, the compliance numbers look much higher because many models produce reasoning without textbook markers.
- Boolean-task compliance is inflated: with only two answers, even a thinking model often lands on the final token quickly.
- Judging whether reduced accuracy under strong no-think prompts is caused by *missing reasoning* or by *parser brittleness* on malformed answers is a known failure mode of the metric suite.

## Sources

- Paper: *Thinking Inertia: LLMs Keep Thinking When Told Not To* — anon, 2026 — [arXiv:2610.11765](https://arxiv.org/abs/2610.11765)
