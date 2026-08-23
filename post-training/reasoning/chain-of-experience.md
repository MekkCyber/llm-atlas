# Chain-of-Experience (CoE)
*Depth — inference-time iterative test-time improvement via accumulated reasoning traces from multi-channel feedback.*

**TL;DR:** Language models can improve within a session by accumulating reasoning traces from **iterative self- and environmental feedback** — **Chain-of-Experience** (CoE). It's the multi-attempt, multi-channel cousin of long single-response chain-of-thought: cheaper per query, more general, directly usable with today's frontier APIs. Aggregate across GPT-5, Gemini-2.5 Pro, Claude-4.5 Sonnet and 8 total models: **+5.6 pp** performance and **−19% API cost** vs. non-CoE baselines. Complementary feedback channels stack; the effect is robust to weak individual feedback signals.

**Prereqs:** [long-cot-rl.md](long-cot-rl.md)
**Related:** [../_post-training.md](../_post-training.md), [../../agents/memory-cognitive-traps.md](../../agents/memory-cognitive-traps.md)

---

## What it is

An inference-time protocol that treats test-time as a **sequence** of related attempts rather than a single response:

- Each attempt produces observable feedback (self-critique, env error trace, judge score).
- Feedback is incorporated into the next attempt's context — either as reflection prompts, extracted lessons, or structured critique.
- Attempts continue until a stopping criterion (correctness signal, budget, self-declared confidence).

The unit of improvement is the **experience chain**, not the single chain-of-thought.

## How it works

**Feedback channels.** The paper studies several:

- **Self-reflection.** The model critiques its own previous attempt.
- **Environmental feedback.** Runtime errors, test failures, tool responses.
- **Judge feedback.** A separate LLM (or the same LLM in a different role) scores attempts.

Channels compose: at each step, one or more feedback types are injected. Compositions of channels give complementary gains — the channels are not substitutable.

**Robustness.** Feedback signals can be weak (noisy self-reflection, partial test coverage, unreliable judges) and the CoE loop still improves over baseline — the model recovers useful signal from noisy sources rather than requiring gold-standard feedback.

## Why it matters

"Test-time compute" has been dominated by longer chains inside a single response. CoE is the multi-attempt, multi-channel axis on the same knob: cheaper (each attempt can be short), more general (works with any frontier API), and directly interoperable with agent memory and inference-time feedback stacks. The +5.6 pp / −19% API cost result is the kind of number that flips agent-serving economics without any training-side change.

## Gotchas & tricks

- Long chains within an attempt and long chains *across* attempts don't just add — the interaction with reasoning-fixation (see MemTrapBench) means naive stacking can hurt. Structure the feedback carefully.
- Judge feedback quality is a load-bearing assumption; a bad judge produces spurious improvements. The paper's robustness result covers *individual weak channels*, not adversarial channels.
- Budget policies matter: without a stopping criterion, CoE can burn arbitrary tokens on plateaued attempts.

## Sources

- Paper: *Chain-of-Experience for Continual LLM Improvement* — Tu, Wang, Xie, Yan (UC Santa Cruz / Bytedance Seed), 2026 — [arXiv:2608.18027](https://arxiv.org/abs/2608.18027)
