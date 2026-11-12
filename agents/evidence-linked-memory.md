# Evidence-Linked Memory (WikiTrace)
*Depth — layered agent memory where terse insights link back to the trajectories that justify them.*

**TL;DR:** Agent "lessons learned" memories often make performance worse — 52% of regressions in the authors' analysis trace back to a bad memory item the agent followed. WikiTrace stores memory in three linked layers — **insights**, **compilations**, and **raw trajectories** — so the agent can walk down from a terse lesson to the evidence that produced it, verify or revise before acting, and skip low-quality lessons entirely.

**Prereqs:** [README.md](./README.md) (agents overview).
**Related:** [README.md](./README.md), [../evaluation/README.md](../evaluation/README.md).

---

## What it is

A design for the external memory an LLM agent accumulates across runs. Instead of flattening past runs into a list of lessons ("prefer tool X for task Y"), WikiTrace stores three levels of structure linked together:

1. **Insights** — one-sentence lessons, surfaced to the agent at every run.
2. **Compilations** — per-source-run summaries explaining *why* each run succeeded or failed; each insight points to the compilations it was derived from.
3. **Raw trajectories** — unaltered execution logs; each compilation points back to its trajectory.

At inference time the agent first sees only insights. When an insight is thin, conflicts with another, or comes from a failed run, the agent is allowed to **descend the links** — retrieve the compilation(s) and trajectory(ies) that back it — and may revise the insight before committing.

## How it works

- **Writing side.** After each run, the agent writes a compilation (what happened, why), then a short insight if a new lesson is warranted. Each insight node stores pointers to its source compilations; each compilation stores a pointer to the raw trajectory.
- **Reading side (at test time).** Agent reads all insights. For each, the agent self-assesses: does it have enough detail, does it conflict with others, is its source a failure? If yes, it retrieves the compilation and / or trajectory and revises its plan.
- **Revisions update insights locally.** Instead of deleting bad lessons, the agent is encouraged to rewrite them in light of the evidence, so the wiki heals over time.

The design mirrors a wiki: a terse surface where most readers stop, with citations and source logs underneath for the moments when terseness isn't enough.

## Why it matters

- **Fixes the dominant memory failure mode.** Memory-induced regression — tasks solved without memory that break with memory — is the practical reason many teams turn off agent memory entirely. WikiTrace drops regression rate from 6.5% (insights-only) to 4.5% while raising mean score by ~3 points.
- **Doesn't need higher-quality insights, just reachable evidence.** The ablation shows the gain comes from the descent mechanism, not from writing cleaner lessons.
- **Scales with task difficulty.** The agent descends more often on harder tasks and gains more there — the structure costs nothing when insights are already sufficient.

## Gotchas & tricks

- **Storage grows without pruning.** Raw trajectories are large; trimming requires policy (keep successful ones? keep failed ones for cautionary evidence?). The paper keeps them all.
- **Descent increases per-run token cost.** Trade some latency for correctness. A production system may cap descents per run.
- **Benchmark signal depends on the backbone.** Gains are replicated across DeepSeek-V4-flash and GLM-5.3-flash; weak backbones may not know when to descend.
- **The lesson-rewrite loop can oscillate** if two insights contradict each other and both get rewritten each pass. Add idempotency / change-rate limits to the rewrite rule.

## Sources

- Paper: *Don't Just Trust Your Memory: Evidence-Linked Memory Reduces Memory-Induced Degradation in LLM Agents* (WikiTrace) — 2026 — [arXiv:2611.14320](https://arxiv.org/abs/2611.14320).
