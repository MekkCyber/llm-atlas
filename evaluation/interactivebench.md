# InteractiveBench — grading MLLMs on guiding a simulated human
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Static multimodal benchmarks (VQA, caption-eval) overstate MLLM readiness for real-world assistance. InteractiveBench puts an MLLM assistant in a **closed loop** with a *scripted human avatar*: the assistant produces natural-language guidance, the simulated human observes, interprets, and acts on their own. **120 multi-turn tasks across six categories**, state-based grading. Frontier MLLMs top out at **20% on Guidance** and **25% on Planning**.

**Prereqs:** [../evaluation/README.md](../evaluation/README.md), [../multimodal/README.md](../multimodal/README.md)
**Related:** [agent-harness-benchmark.md](./agent-harness-benchmark.md), [../agents/README.md](../agents/README.md)

---

## What it is

A simulation environment where the MLLM assistant and a scripted "simulated human" each have their own observation and action streams. The assistant sees the scene and must produce natural-language guidance that the simulated human (another agent, with its own policy) can act on. The human interprets instructions and executes actions; grading is based on whether the human reaches the target state.

The benchmark covers six task categories, including guidance, planning, and recovery scenarios, with 120 multi-turn tasks total.

## How it works

- **Two-agent loop.** Assistant and simulated human exchange messages/actions in a shared environment. The assistant never acts directly; it only speaks.
- **State-based graders.** Each task has an executable goal state (e.g. "object X is in location Y"); the terminal state determines pass/fail. No LLM judge.
- **Scoring axes.** Guidance (the assistant produces actionable instructions), Planning (the assistant decomposes the task correctly), with per-category breakdowns.
- **Simulated human.** Not an open-ended free agent — it is scripted with a reasonable interpretation policy and action primitives. This isolates "the model's guidance was wrong" from "the human did something weird."

## Why it matters

Benchmarks like VQA and captioning are solved on their own distributions, which overpromises on physical-world assistance. InteractiveBench exposes a sharp gap: 12 frontier MLLMs max out at **20% on Guidance** and **25% on Planning**. The pattern — the model knows what to do in principle but can't produce instructions another agent can follow — is exactly the one that matters for AR glasses, teleguidance, and multi-agent assistants.

## Gotchas & tricks

- **"Simulated" is doing work.** Numbers depend on the simulated-human policy; a slightly different interpretation model would give slightly different rankings.
- **Six-category breakdown matters.** Different models fail on different categories; the aggregate top-line understates the variance.
- **Not a substitute for human-in-the-loop evaluations** — it is cheaper and reproducible, but a scripted human cannot fully substitute for a real one.
- **Pairs naturally with [agent-harness-benchmark.md](./agent-harness-benchmark.md).** Guidance quality is model; harness quality is runtime; both should be scored separately.

## Sources

- Paper: *InteractiveWorld: A Verifiable Human-Centric Environment for Building Better Interactive MLLM Assistants* — Wang et al., Meta (with Duke), 2026 — [arXiv:2610.33966](https://arxiv.org/abs/2610.33966).
