# Experiential Learning (LLM-as-a-Coach)

*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Scalar-reward RL on non-verifiable tasks (creative writing, dialogue, open-ended assistance) compresses rich rubric feedback into a number, discarding the specifics of *why* a response is good or bad. Experiential Learning (EL) instead has the judge model act as a *coach*: it distills its own critique of each on-policy response into transferable textual "experiential knowledge," which conditions a teacher model to produce a target trajectory. The student then internalizes that trajectory via on-policy context distillation. The reward signal is now text, not a scalar — much higher bandwidth, and harder to hack.

**Prereqs:** [cot-reward-model.md](./cot-reward-model.md), [_rewards.md](./_rewards.md)
**Related:** [dpo.md](./dpo.md) · [rejection-sampling.md](./rejection-sampling.md) · [_post-training.md](./_post-training.md)

---

## What it is

A three-role training loop:

- **Policy** (the student, on-policy generator).
- **Coach** (an LLM judge that produces critique — the "experiential knowledge").
- **Teacher** (an LLM conditioned on the coach's critique that generates the target trajectory).

The student distills from the teacher on-policy. The reward channel is the coach's textual critique, not a scalar.

## How it works

For each training batch:

1. The policy generates response $o$ to prompt $q$.
2. The coach judges $o$ against the rubric and distills the judgment into a compact textual "experience" $e$ (e.g., "the response leans on cliché in paragraph 2; try grounding claims in specifics").
3. The teacher receives $(q, e)$ and generates a target response $o^*$ that reflects the experience.
4. The student is updated to reduce $\mathrm{KL}(\pi_\text{student}(\cdot \mid q) \| \pi_\text{teacher}(\cdot \mid q, e))$ — standard context-distillation loss, but with the experience folded into the teacher's condition.

Because the coach and teacher can be the *same* model as the policy (or a proprietary model with critique capability), no separate reward-model training is needed.

## Why it matters

Rubric-based scalar RL is the state of the art for non-verifiable tasks, and its brittleness is widely acknowledged: reward hacking, mode collapse toward one "safe" style, insensitivity to fine distinctions among high-quality responses. Textual experiential feedback preserves those distinctions and gives the student something specific to fix. The paper reports EL beating rubric-based RL on held-out *and* unseen open-ended tasks across two policy families, with better OOD generalization and reduced reward hacking.

## Gotchas & tricks

- The experience $e$ must be *transferable*, not just a critique of the current response. Prompts that ask the coach for "actionable general lessons" beat prompts that ask for "specific corrections."
- Context distillation requires the teacher's target to be *reachable* by the student's on-policy distribution. Too-far teachers still cause the usual OPD problems.
- The coach can be the same model as the policy — this is the more scalable regime and works well enough in practice; a stronger proprietary coach is a further-improvement lever.

## Sources

- Paper: *LLM-as-a-Coach: Experiential Learning for Non-Verifiable Tasks* — authors not shown on HF page, 2026 — [arXiv:2607.18110](https://arxiv.org/abs/2607.18110) · [HF](https://huggingface.co/papers/2607.18110)
