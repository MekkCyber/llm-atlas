# SWE-Together
*Depth — one specific benchmark, grounded in its source paper(s).*

**TL;DR:** A multi-turn coding-agent benchmark reconstructed from **real user–agent sessions**: 109 repository-level tasks curated from 11,260 recorded sessions where the user's intent and outcome are recoverable. Replayed against candidate agents via a **reactive** LLM user-simulator that intervenes only when the agent stalls. Grades both final repository correctness *and* number of corrective feedback turns.

**Prereqs:** [README.md](./README.md), [livecodebench.md](./livecodebench.md)
**Related:** [tua-bench.md](./tua-bench.md), [osworld2.md](./osworld2.md), [../agents/README.md](../agents/README.md)

---

## What it is

SWE-bench-style evaluation gives an agent the full spec up-front and grades the final code. Real IDE/CLI coding is *interactive* — users clarify, correct, and add constraints turn-by-turn. SWE-Together captures that shape.

## How it works

**Task construction.**
- Mine 11,260 recorded user–agent coding sessions.
- Filter for sessions with (a) recoverable initial repo state, (b) reconstructible clear user goal, (c) observable outcomes (tests, produced artifacts, user confirmation).
- Retain 109 repository-level tasks.

**Reactive user simulator.**
- An LLM plays the original user, conditioned on the recovered intent.
- **Reactive**, not proactive: it only messages the coding agent when the agent's own progress signals it's stuck (asking for clarification, wandering off-spec, waiting for input). This keeps turn counts honest — no trivial-turn inflation.

**Metrics.**
- **Final correctness** on repository-level outcomes.
- **Interventions required** — how many corrective turns the simulator issued. Fewer is better.

**Frontier signal.** Stronger frontier agents show **both** higher success *and* fewer interventions — the two axes are correlated. This makes SWE-Together more discriminating than static coding benchmarks that only report final pass rates.

## Why it matters

- **Matches real deployment.** Claude Code, Cursor, and Codex operate in exactly this reactive-turn shape. Benchmarks that ignore it under-measure UX quality.
- **Collaboration cost as a metric.** "Turns to unblock" surfaces friction that final-code grading misses.
- **Real-session grounding.** Session mining beats synthetic task authoring for ecological validity — the tasks come from tasks users actually attempted.

## Gotchas & tricks

- **Simulator ≠ real user.** The reactive LLM is a proxy; agents that game the simulator's failure detector could inflate success. Cross-check on a small human-in-the-loop slice.
- **109 tasks is a small pool.** Fine for a headline metric; noisier for per-language or per-framework breakdowns.
- **Requires session-level recording infrastructure.** Reproducing the task-construction pipeline needs consented, structured session logs — few labs have this.

## Sources

- Paper: *SWE-Together: Evaluating Coding Agents in Interactive User Sessions* — Zhao et al. (Meta), 2026 — [arXiv:2606.29957](https://arxiv.org/abs/2606.29957).
