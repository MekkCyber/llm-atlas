# FinCosmos
*Depth — an open-world agent benchmark where correctness is checked procedurally, with no reference answer.*

**TL;DR:** A financial-analysis benchmark where a simulated user holds a hidden strategy and reveals it only when the agent asks. The agent must fetch data from the live web, build its own backtest without look-ahead, and deliver verifiable results. No oracle answer is needed — grading uses 15 veto rules on data provenance, point-in-time correctness, and whether the reported metrics can be recomputed from the delivered trade log. Across 150 sessions on 10 frozen tasks, **zero** acceptances by current agents.

**Prereqs:** [README.md](./README.md) (evaluation overview), [../agents/README.md](../agents/README.md).
**Related:** [README.md](./README.md), [../agents/README.md](../agents/README.md).

---

## What it is

An evaluation framework for analysis-and-reporting agents. Three design choices make it "open-world":

- **Hidden objective.** The user holds a strategy sampled from a compositional grammar (~3.8M combinations) and only reveals pieces when the agent asks — the agent must drive clarification itself.
- **Unknown data source.** No dataset is handed to the agent. It fetches from the web under a cutoff date, and the fetch must leave a provenance trail.
- **No reference answer.** Grading is procedural. An accept requires both (a) the simulated user signs off and (b) none of 15 veto rules fires.

A session is a single task attempt from clarification through final report.

## How it works

The 15 veto rules check:

- **Provenance** — every data point cited in the report must come from a documented source with a URL, a retrieval timestamp, and a cutoff respected.
- **Point-in-time correctness** — no row in the backtest may use information after its own timestamp. Look-ahead bias is detected by replaying the backtest.
- **Recomputability** — the reported Sharpe, drawdown, and total return must be reproducible from the delivered trade log alone, within a tolerance.
- **Strategy match** — the realized trades implement the user's hidden strategy, not a different one the agent guessed and backfit to.

Graders are programs, not LLMs, so grading is deterministic and cheap. New tasks don't require labels — they only require encoding the user's strategy into the grammar.

## Why it matters

- **Silent-failure diagnosis.** Agents produce reports that *look* convincing but are wrong on a specific, reproducible axis (data from after the cutoff; unauthorized strategy; non-recomputable metrics). FinCosmos names and measures each axis.
- **Current agents fail completely.** 4 coding agents × 1 no-tools model × 150 sessions on 10 tasks → 0 accepts. Agents raised only 54–64% of necessary clarifying questions; human experts attributed 88% of failures to the agents, not the framework.
- **One honest route forward.** An agent with persistent memory eventually solved all 10 tasks after multiple attempts — persistent self-correction from procedural feedback is tractable; one-shot is not.
- **Transferable rubric.** Provenance + point-in-time + recomputability generalizes to any analysis-report workload. Finance is just the first domain where the whole stack can be mechanically checked.

## Gotchas & tricks

- **Compositional grammar size matters.** With 3.8M strategies, memorization is infeasible — a model that passes a task really did the work for that task.
- **Program checks are strict.** A single after-cutoff citation vetoes the session; this is the point, but it also means harshness is a feature, not a tuning knob.
- **Benchmark is cutoff-aware.** All tasks are frozen with a fixed data cutoff; the agent cannot reach future data even if the web has it.
- **Sign-off is necessary but not sufficient.** The user signs off based on apparent plausibility; the veto rules catch what the user misses.

## Sources

- Paper: *FINCOSMOS: An Open-World Evaluation Paradigm for Agents in Finance* — Peng, Qian, Li, Yu et al., 2026 — [arXiv:2611.07464](https://arxiv.org/abs/2611.07464).
