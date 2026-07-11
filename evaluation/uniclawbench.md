# UniClawBench
*Depth — a capability-driven, live-Docker benchmark for proactive agents.*

**TL;DR:** 400 bilingual real-world agentic tasks organized by five **base capabilities** (skill usage, exploration, long-context reasoning, multimodal understanding, cross-platform coordination) rather than by scenario. Executes in live Docker containers with fine-grained step-by-step checkpoints, and grades via a closed-loop three-agent setup (executor + hidden supervisor + user simulator). Introduced by HKU MMLab + Meituan (2026).

**Prereqs:** *(none)*
**Related:** [../agents/README.md](../agents/README.md), [ifeval.md](./ifeval.md), [livecodebench.md](./livecodebench.md)

---

## What it is

An agent-evaluation benchmark whose two design commitments break with prior work:

1. **Capability-first taxonomy.** Tasks are grouped by which base-model capability they test — not by application domain. This makes it possible to attribute an agent failure to a specific capability, not a bundle of them. Existing scenario-based benchmarks (WebArena, OSWorld, τ²-Bench) confound multiple capabilities per task, making root-cause analysis hard.
2. **Live-Docker execution with step checkpoints.** Instead of comparing a final answer to a static gold string, tasks run in live Docker containers and are graded against **fine-grained checkpoints** interleaved with execution. This handles multi-solution paths (many valid ways to end up correct) and detects partial progress.

## How it works

**Task construction.** 400 bilingual tasks, each labeled with the base capability under test. The five capabilities:

- **Skill Usage** — invoking the right tool with the right arguments in the right order.
- **Exploration** — discovering what tools / environments / information are available before committing.
- **Long-Context Reasoning** — integrating state across long trajectories.
- **Multimodal Understanding** — combining vision/text/tabular signals.
- **Cross-Platform Coordination** — moving state between separate applications.

**Grading loop.** Three agents:

- **Executor agent** — the model under test, running its native agent framework.
- **Hidden supervisor agent** — grades progress against the step checkpoints. Hidden from the executor so it cannot condition its actions on grading criteria.
- **User agent** — simulates a realistic user, providing multi-turn clarifications and follow-ups. Also hides its underlying rubric.

**Framework decoupling.** Every model is evaluated under multiple agent frameworks, so the paper can decompose leaderboard rankings into "base model capability" vs "framework glue quality" — the two axes that usually get conflated.

## Why it matters

- **Attribution over aggregation.** Capability-decomposed scores tell you which capability to work on; scenario-decomposed scores just tell you the field's leaderboard.
- **Live-Docker checkpoints.** Dynamic environments detect partial progress and reward shorter valid solutions — necessary for evaluating exploration and cross-platform tasks.
- **Framework-vs-model decomposition.** Anyone building agent frameworks on top of frontier models needs this decomposition to know whether their tweaks are helping.

## Gotchas & tricks

- **Docker overhead.** Live containers are more expensive than static rubrics; the benchmark isn't cheap to run at scale.
- **User-agent bias.** Whatever LLM plays the user shapes the difficulty; results depend on that model too, though the closed-loop design tries to hide grading criteria.
- **Capability boundaries are fuzzy.** "Long-context reasoning" and "exploration" overlap in some tasks; the capability label is the *dominant* test, not an exclusive one.
- **Bilingual coverage.** 400 bilingual tasks is a modest scale; multi-language coverage beyond the two languages here is a follow-on.

## Sources

- Paper: *UniClawBench: A Universal Benchmark for Proactive Agents on Real-World Tasks* — Chen et al., HKU MMLab + Meituan, 2026 — https://arxiv.org/abs/2607.08768
- Code: https://github.com/HKU-MMLab/UniClawBench
