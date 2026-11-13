# Interactive Visual Gym
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A closed-loop evaluation and training framework for vision-language models that act as *interactive assistants* — cooking, household manipulation, mobile-UI help. Instead of single-turn Q&A, the VLM converses with a **persona-driven simulated user** inside a task simulator that provides ground-truth success signals. The paper's IVG-Bench (315 scenarios) shows top models complete only 53.4% of tasks in time, and a GRPO proof-of-concept improves a 4B VLM.

**Prereqs:** [README.md](README.md)
**Related:** [../multimodal/README.md](../multimodal/README.md), [../post-training/grpo.md](../post-training/grpo.md)

---

## What it is

Benchmarks for VLMs have mostly measured single-turn captioning, VQA, or static grounding. Real deployments look very different: the user is doing something (cooking, phone navigation, troubleshooting), may be wrong, and asks follow-up questions. Interactive Visual Gym (IVG) builds a harness around that regime.

The harness combines two things not usually put together: a task *simulator* that knows the correct answer at every step, and a *simulated user* driven by an LLM persona that embodies a skill level, misconceptions, and goals.

## How it works

Three components:

1. **Task simulators.** Procedural environments for cooking, household manipulation, and mobile UI flows. The simulator knows ground-truth state, exposes an observable render, and emits a success signal when the user reaches the goal.
2. **Persona-driven simulated users.** LLM agents conditioned on a persona spec (e.g. "beginner cook, impatient, misreads labels") that interact with the VLM in natural language. They make realistic mistakes and ask follow-ups, so the interaction trace is varied without human raters.
3. **Metrics.** Both **task success** (from the simulator) and **interaction quality** (groundedness in the visible scene, actionability of the step, recovery from user error). Human-study agreement with the interaction quality metric is 78.5%.

For training, the simulator's success signal is a reward usable in RL; the paper shows GRPO already works on a 4B VLM.

## Why it matters

- Gives a closed-loop *and* RL-ready bench for interactive VLMs, where before there were only static benchmarks and expensive human evals.
- The 53.4% top-model success rate leaves a lot of room: this is a regime far from saturated.
- Surfaces the two failure modes that single-turn benchmarks miss: factual grounding on the visible scene, and giving a *next step* rather than a description.

## Gotchas & tricks

- Persona quality is critical — generic LLM users converge to over-helpful behavior and inflate success. Keep personas narrow and consistent.
- Interaction-quality metrics are LLM-judge based; sanity-check per task that the judge isn't rewarding verbosity.
- GRPO signal is noisy because the simulated user is also an LLM; freeze the user policy during RL on the assistant.
- Not a substitute for real-user evals, but a cheap filter before them.

## Sources

- Paper: *Interactive Visual Gym: Evaluating Visual Interaction Models with Simulated Users in Interactive Worlds* — Ma, Li, Ranasinghe, Zhang, Panagopoulou, Huang, Shangguan, Tan, Savarese, Niebles, Krishna, Xu, Zhou — Salesforce AI Research (2026) — [arXiv:2611.08286](https://arxiv.org/abs/2611.08286)
