# WebArena-Pro
*Depth — 300-task, 20-app, multimodal, isolated-container benchmark for web agents.*

**TL;DR:** A reproducible web-agent benchmark that fixes three gaps in prior WebArena-style suites: audio/video treated as first-class observations, every task run in a seeded isolated container (so parallel evaluation is actually reproducible), and all 300 tasks human-authored through an annotator UI with LLM triage and human validation. Top configurations (Codex w/ GPT 6.1 Sol, Claude Code w/ Claude Opus 5.5) reach ~62.3%; best GenericAgent reaches 30.3%.

**Prereqs:** [./README.md](./README.md)
**Related:** [../agents/README.md](../agents/README.md)

---

## What it is

A benchmark for evaluating LLM agents that operate inside real web applications. 300 human-authored tasks over 20 self-hosted apps (shopping, forums, dev tools, email, maps, media, …) across six domain categories. Designed to be run end-to-end by any agent that can observe browser state and emit actions.

## How it works

**Observation space.**
- DOM + screenshot as always.
- **Audio** and **video** promoted to first-class observations. New actions: transcribe audio, extract video frames, and — for omni models — pass raw media directly to the LLM.

**Reproducibility.** Each task boots its own seeded Docker container with a fresh snapshot of the target app, so two runs are byte-identical before the agent acts. Parallel runs don't interfere; nightly re-runs give stable scores.

**Authoring pipeline.** Tasks are written by human annotators through a purpose-built UI, filtered by LLM-assisted triage (de-dup, difficulty check), and human-validated for completability and ground-truth action traces.

**Scoring.** Each task has a programmatic success check (DB state, URL, form submission). Binary success per task, averaged into a suite score.

## Why it matters

WebArena became a standard benchmark but had known issues: parallel runs weren't reproducible, media observations were out-of-scope, and some tasks were ambiguous. WebArena-Pro raises the floor on all three. The reported 62.3% / 30.3% gap between frontier coding-agent scaffolds and generic agents is a clean signal that *scaffold design* carries as much weight on web as base-model capability.

## Gotchas & tricks

- **Self-hosted apps** — avoids the fragility of real external sites (A/B tests, auth, rate limits) but lags real web complexity. WebArena-Pro bets on reproducibility over realism.
- **Parallel safety.** Only true because every task is containerized and seeded; shared-state benchmarks silently race.
- **Omni-model actions.** Agents that can pass raw media to the LLM skip transcription and frame extraction; the benchmark supports both, which lets you isolate the gain from omni inputs.
- **Scaffold vs model.** The large gap between Codex/Claude-Code harnesses and GenericAgent is the headline — scaffolds move numbers as much as switching to a stronger base.

## Sources

- Paper: *WebArena-Pro: A Heterogeneous, Multimodal, Reproducible Benchmark for Web Agents* — Kerboua et al. (McGill / Mila / Ohio State / ServiceNow), 2026 — [arXiv:2611.17955](https://arxiv.org/abs/2611.17955)
