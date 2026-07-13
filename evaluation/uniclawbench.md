# UniClawBench
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A **capability-driven** benchmark for proactive LLM agents on **real-world tasks**, evaluated inside **live Docker containers** with step-by-step completion checkpoints. **400 bilingual** tasks, decomposed by five foundational capabilities (Skill Usage, Exploration, Long-Context Reasoning, Multimodal Understanding, Cross-Platform Coordination) rather than by scenario. A **closed-loop harness** — executor agent + hidden supervisor + user agent — simulates realistic multi-turn feedback without leaking grading criteria. Explicitly designed to distinguish base-model capability from framework design. Introduced by Chen et al. (HKU MMLab / Meituan), 2026 (arXiv 2607.08768).

**Prereqs:** *(none)*
**Related:** [causalds.md](./causalds.md) · [ifeval.md](./ifeval.md) · [mmlu.md](./mmlu.md)

---

## What it is

An agent benchmark that fixes three problems with prior work.

**Problem 1 — capability confounding.** Scenario-based tasks ("book a flight," "debug this repo") test many capabilities at once. When an agent fails, you can't tell whether it was planning, tool use, or multimodal parsing that broke. UniClawBench decomposes tasks by *underlying capability* into five categories:

- **Skill Usage** — knowing which tool to reach for.
- **Exploration** — probing the environment when unsure.
- **Long-Context Reasoning** — carrying earlier state through a long trajectory.
- **Multimodal Understanding** — vision, screenshots, mixed inputs.
- **Cross-Platform Coordination** — moving state across tools/apps/OS boundaries.

Failures are attributable, not just aggregated.

**Problem 2 — sandbox evaluation.** Prior benchmarks pre-record answers or run in restricted environments that don't reflect production. UniClawBench spins up **live Docker containers** per task and grades via **fine-grained checkpoints** that inspect intermediate state — not just the final answer.

**Problem 3 — evaluation contamination.** Multi-turn benchmarks that expose the grading criteria to the agent leak signal. UniClawBench's **closed-loop harness** has three roles: an **executor agent** (the model under test), a **hidden supervisor agent** (grades against secret criteria the executor never sees), and a **user agent** (simulates realistic human feedback). The executor gets multi-turn signal, but not the rubric.

**Scope.** 400 bilingual tasks. Public benchmark and code at https://github.com/HKU-MMLab/UniClawBench.

## Why it matters

- **Agent leaderboards have been running ahead of production performance** for years — models pass benchmarks but fail on real tasks. UniClawBench's live-container + hidden-supervisor design is a plausible template for evals that predict production behavior.
- **Capability decomposition changes model comparison.** Two models with the same average score can have opposite capability profiles; UniClawBench surfaces this instead of averaging it away. Useful for choosing base models for specific agent products.
- **Separates base model from harness.** The paper deliberately runs each model under multiple agent frameworks and finds base capabilities and framework design *jointly* shape performance — neither dominates. Concrete guidance for anyone building a stack.

## Gotchas & tricks

- **Docker overhead.** Live containers per task make evaluation expensive. Batching and container reuse are practical necessities for iteration speed.
- **Hidden supervisor is a design commitment.** If the supervisor's grading criteria drift, historical comparisons become invalid. Version the supervisor as carefully as the benchmark.
- **Bilingual scope** (English + Chinese) is deliberate but limited. Agent behavior varies by language for reasons unrelated to capability (data mix, tokenizer, training coverage); reading UniClawBench numbers cross-language requires care.
- **Framework contamination.** If your model's post-training already saw traces from a specific agent framework, comparing "same model under two frameworks" isn't clean. The paper acknowledges this.

## Sources

- Paper: *UniClawBench: A Universal Benchmark for Proactive Agents on Real-World Tasks* — Chen, Duan, Sun, Li, Wang, Zhang, Liu (HKU MMLab / Meituan), 2026 — [arXiv 2607.08768](https://arxiv.org/abs/2607.08768).
- Code / benchmark: https://github.com/HKU-MMLab/UniClawBench
- Related evals: Terminal-Bench 2.0, τ²-Bench, SWE-bench, WebArena — sibling agent benchmarks.
