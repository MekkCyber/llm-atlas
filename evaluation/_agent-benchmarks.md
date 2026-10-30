# Agent Benchmarks

*Taxonomy — benchmarks for LLM agents: tool-using, computer-use, CLI, browser, and multi-turn conversational.*

**TL;DR:** Agent benchmarks have split along three independent axes: **surface** (CLI, browser, OS, tool-API, chat), **grading signal** (outcome, procedure, collaboration, user-variation), and **task source** (synthetic, scraped, hand-authored). The 2024-era canonical set (SWE-bench, WebArena, OSWorld, TauBench) measures outcome under full-spec prompts. The 2026-era set introduced in this day's digest adds *user-variation* (PersonaCUABench), *collaboration quality* (RealCLI), and *procedure compliance* (TauIndianBankBench). The modern default is **report multiple benchmarks across the three axes**, since the signals diverge.

**Related taxonomies:** [_optimizers](../pre-training/_lr-schedules.md)
**Depth files covered here:** [cli-agent-bench](cli-agent-bench.md) · [persona-cua-bench](persona-cua-bench.md) · [tau-indian-bank-bench](tau-indian-bank-bench.md) · [instruction-capacity](instruction-capacity.md)

---

## The problem

LLM agents are evaluated on tasks that look like "do this complex multi-step thing." But a single scalar score (did the task complete?) hides at least four distinct quality axes:

- Can the agent complete the task **at all**? (capability)
- Does it comply with **required procedures**? (regulated / audited settings)
- Does it work **with** the user in-run (collaboration)?
- Is it robust to **user variation** (persona / phrasing drift)?

An agent benchmark that collapses all four into one number misleads deployers. Modern benchmarks instead measure one axis each.

## The shared pattern

Every agent benchmark has three components:

```
tasks:   a corpus of ground-truth-labeled scenarios
env:     a simulated or sandboxed environment the agent acts in
graders: automated checks (+ optionally LLM-judge or human spot-checks)
```

What differs is the **axis** each grader measures. The axis determines the signal the benchmark provides.

---

## Variants

| Benchmark | Surface | Primary axis | When it wins |
| --- | --- | --- | --- |
| SWE-bench (2023) | code repo + shell | outcome (patch correctness) | measuring code-agent capability on real repos |
| WebArena (2023) | browser | outcome (final-state match) | web navigation / multi-step browsing |
| OSWorld (2024) | desktop (Ubuntu/Windows) | outcome (final-state match) | general CUA capability |
| Windows Agent Arena (2024) | desktop (Windows) | outcome | Windows-specific CUA eval |
| TauBench (2024) | tool API + user agent | outcome + light procedure | multi-turn tool-use chatbot eval |
| Terminal-Bench (2024) | CLI shell | outcome | CLI-agent capability |
| [cli-agent-bench](cli-agent-bench.md) RealCLI (2026) | CLI shell | outcome + 5-dim **collaboration** | measuring whether CLI agents work with the user |
| [persona-cua-bench](persona-cua-bench.md) (2026) | desktop CUA | outcome + user-satisfaction, under **44 personas** | robustness to user variation |
| [tau-indian-bank-bench](tau-indian-bank-bench.md) (2026) | banking API | outcome + **procedure compliance** + non-intervention + audit | regulated enterprise agents |
| [instruction-capacity](instruction-capacity.md) (2026) | plain text | concurrent-instruction load curve | system-prompt pressure for agents and chat |

---

## How to choose

- **Capability-first screening**: SWE-bench (code), OSWorld (desktop), WebArena (web), Terminal-Bench (CLI). Use the one matching your deployment surface.
- **User-variation robustness**: PersonaCUABench. Required if your deployment surface is consumer-facing and real users will phrase things in their own voice.
- **Collaboration quality**: RealCLI (CLI) or TauBench (API) with the collaboration rubric. Required if the agent will run in a human-in-the-loop setting with mid-run user input.
- **Procedure compliance**: TauIndianBankBench or a domain-specific equivalent. Required for regulated enterprise deployment (banking, healthcare, legal).
- **System-prompt load**: Instruction Capacity. Required if the agent will carry heavy system prompts (many tool schemas, many role rules).

Default policy: **always run at least one capability benchmark on your surface, plus one quality benchmark (collaboration, user-variation, or procedure) matched to your deployment setting.** A single number ships a hidden weakness.

---

## Adjacent but distinct

- **Reasoning benchmarks** ([aime](aime.md), [math500](math500.md), [livecodebench](livecodebench.md)) — single-turn, no tool use. Measure the backbone, not the agent loop.
- **Instruction-following** ([ifeval](ifeval.md)) — measures *one* instruction's correctness. Instruction Capacity is the concurrent-load generalization.
- **Safety evals** ([safety/](../safety/)) — adversarial behavior under agentic settings is a different axis; RealCLI's "safety-boundary management" dimension overlaps.

---

## Sources

- Paper: *SWE-bench* — Jimenez et al., 2023.
- Paper: *WebArena* — Zhou et al., 2023.
- Paper: *OSWorld* — Xie et al., 2024.
- Paper: *TauBench* — 2024.
- Paper: *RealCLI* — Liu et al., 2026, [arXiv 2610.22507](https://arxiv.org/abs/2610.22507).
- Paper: *PersonaCUABench* — Li et al., 2026, [arXiv 2610.22990](https://arxiv.org/abs/2610.22990).
- Paper: *TauIndianBankBench* — Adhikary & Devadiga, NPCI, 2026, [arXiv 2610.30161](https://arxiv.org/abs/2610.30161).
- Paper: *Instruction Capacity* — Kutomanov, 2026, [arXiv 2610.36095](https://arxiv.org/abs/2610.36095).
