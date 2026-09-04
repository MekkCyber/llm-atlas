# Agent Harness

*Depth — the model-external execution infrastructure around an LLM agent, treated as a first-class artifact that can be created, evolved, and evaluated.*

**TL;DR:** An agent's capability depends on two things: the model weights, and the *harness* — the code that manages context, control flow, tool routing, retries, and memory. Changing the harness while holding the model fixed can shift task performance significantly, but most benchmarks report scores under an implicit, human-engineered harness. **HarnessDev** (Wu et al., 2026) reframes evaluation to measure the harness itself: a **Creation** stage (build a runnable harness from a minimal seed) and an **Evolution** stage (iteratively revise it from downstream feedback), graded on capability *and* execution-token cost across held-out benchmarks.

**Prereqs:** [README](README.md), [../evaluation/README](../evaluation/README.md)
**Related:** [../post-training/_rl](../post-training/_rl.md)

---

## What it is

The **harness** is the executable substrate an LLM operates inside. It includes:

- **Context management** — what goes into the prompt each turn (system prompt, tool schemas, memory / RAG snippets, prior tool outputs, truncation policy).
- **Control flow** — tool loops, retries, timeouts, fallback branches, stop conditions.
- **Tool routing** — which tool schema is exposed when, how outputs are parsed, error handling.
- **Memory** — scratchpads, episodic memory, long-lived state across turns.
- **Cost governance** — token budgets, step caps, model-tier routing.

Two agents that share weights but differ in harness can produce very different downstream numbers. A harness is *code* — versionable, testable, and — as HarnessDev formalizes — optimizable in its own right.

## How it works

### HarnessDev's evaluation protocol

Two stages, both scored on **capability** (task success on held-out benchmarks) and **efficiency** (execution-token cost):

**Creation.** Agent receives a *minimal seed harness* and a small number of example cases, then must build a complete execution system from scratch. Six creator LLMs evaluated across four domains (code, search-and-research, writing, ML experimentation) and five downstream benchmarks totaling 2,207 unique instances. Hidden evaluation tasks are withheld from development.

**Evolution.** Agent starts from its own Creation-stage harness and iteratively revises it using downstream execution feedback, targeting benchmark performance improvement.

Both stages produce runnable code artifacts, not just prompts.

### The Creation vs Evolution split matters

Creation isolates whether the model can *design* a scaffold from scratch. Evolution isolates whether the model can *improve* one given feedback. HarnessDev finds these are separable capabilities: some models create decent harnesses but fail to evolve them meaningfully, and vice versa.

### Reported empirics

- Generated harnesses stay **substantially below** mature human-engineered references on code and search-and-research.
- They match or exceed references on writing and ML experimentation (domains with weaker human baselines).
- **Wide variance in execution cost** — a matched-capability harness can be 2–5× more expensive on tokens.
- **Evolution gains are unstable** — improvements on the training set often don't transfer to held-out tasks.
- Gains **depend strongly on the runtime model** — a harness that works for one executor model may not for another.

## Why it matters

- **Reframes "agent performance" as two decoupled axes.** Every reported benchmark number today conflates weights and scaffold; HarnessDev shows how much of the variance is scaffold-side.
- **Puts scaffold-authoring on the roadmap.** If self-improving agents are the frontier, they need to author their own scaffolds — HarnessDev is the first benchmark that grades that ability separately from task solving.
- **Cost is a first-class axis.** Two harnesses at the same accuracy can differ 5× in token cost; the benchmark reports both, breaking the accuracy-only leaderboard pattern.
- **Enables joint weight+harness optimization.** Companion work like WHALE (Weight-Harness Alternating LEarning) alternates model updates with harness updates; HarnessDev provides the evaluation substrate.

## Gotchas & tricks

- **Choose the runtime model deliberately.** HarnessDev shows harness quality doesn't transfer across executors. Optimizing a harness on one model and deploying it on another is not safe by default — re-evaluate.
- **Beware Evolution overfit.** Iterative harness edits driven by downstream feedback are essentially SGD on the training set; hold out real tasks or the harness will overfit its edit loop.
- **A short seed matters.** Too rich a seed harness contaminates Creation (the model just tweaks a mature scaffold); too skeletal a seed prevents runnable output. HarnessDev's seed is deliberately minimal.
- **Report cost per task, not per benchmark.** A harness that halves the token cost on easy tasks but 3×'s it on hard ones can average to "similar cost" while being much worse to operate.
- **Don't conflate harness with prompt.** The prompt is one component of the harness; joint-adaptation methods that only optimize prompts leave the broader harness fixed.
- **Human references are a moving target.** The strongest human-engineered harness on a benchmark today may be different from six months ago (SWE-agent, AutoGPT, etc.). Version the reference.

## Sources

- Paper: *HarnessDev: Can LLMs Create and Evolve Their Own Agent Harness?* — Yuhao Wu, Jingyuan Zhang, Jiajun Shi et al. — 2026 — [arXiv:2609.01437](https://arxiv.org/abs/2609.01437) — ByteDance Seed · SUTD · Georgia Tech · M-A-P · TokenWave.AI.
- Related: *WHALE: A Simple Recipe for Joint Harness-Weight Optimization* — Kim et al., 2026 — [arXiv:2609.00196](https://arxiv.org/abs/2609.00196).
- Related: *Aspire: Can Models Self-Evolve from Vague Goals?* — Wu et al., 2026 — [arXiv:2608.31111](https://arxiv.org/abs/2608.31111).
