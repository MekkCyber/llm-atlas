# FM-Bench
*Depth — the 20-season football-management benchmark for long-horizon LLM agents.*

**TL;DR:** FM-Bench evaluates whether an LLM agent stays effective across **20 simulated football-management seasons** (~340–400 decision points, 26 tools). Two tracks: **Solo** (agent vs the simulation) and **Arena** (agent vs competing agents). Introduced by Wang et al., 2026. Fifteen frontier models were run; all completed the horizon, but the winners were distinguished by **managerial behaviour**, not raw computation.

**Prereqs:** *(none)*
**Related:** [ifeval.md](./ifeval.md), [../agents/README.md](../agents/README.md)

---

## What it is

Paper: *FM-Bench: A Benchmark for Long-Horizon Management with Competing Agents*, Wang et al., 2026, arXiv 2608.18423.

- **Domain:** a football-club management simulator with a persistent economy, roster, contracts, facility investments, and a competitive league.
- **Horizon:** **20 in-simulation seasons**, ~340–400 discrete decision points per run.
- **Tool set:** **26 tools** covering drafting, trading, contract negotiation, facility investment, tactics, and lineup management.
- **Tracks:**
  - **Solo** — agent plays against the simulation only.
  - **Arena** — multiple agents share a league and compete for finite talent / prize money.
- **Metrics:** long-run club value, silverware, and normalised head-to-head win rate; plus behavioural probes (investment trajectory, negotiation initiation rate).

## How it works as an LLM eval

- The agent interacts turn-by-turn via a JSON tool interface; observations are structured summaries plus recent-events text.
- A scripted rule-based baseline plays alongside so absolute scores are anchored.
- Runs are single-seed by default; the paper repeats seeds per model to give error bars on long-run outcomes.

## Why it matters

- **Long-horizon signal is scarce.** Existing agent benchmarks tend to score a single task horizon. FM-Bench forces plans that span *simulated years* and rewards behaviours (long-term investment, roster planning) invisible on short horizons.
- **Managerial-behaviour axis.** All 15 frontier models finish; separation comes from qualitative behaviours — reducing long-term investment as the horizon ends, preserving cash, initiating negotiations proactively. Gives the field a way to grade *strategic* competence, not just tool-call accuracy.
- **Competitive multi-agent tier.** The Arena track is one of the few standardised long-horizon *adversarial* agent evals.

## Gotchas & tricks

- **Domain saturation.** Frontier models with football-league priors may benefit from unrelated pretraining data; the paper reports minor prior effects but doesn't fully de-confound.
- **Cost.** 340–400 decisions × several LLM calls each × 20 seasons × arena opponents runs into millions of tokens per full evaluation. Plan compute accordingly.
- **Scripted baseline is not "human-level".** It's a control, not an oracle — a large gap between LLM and script is a floor, not a ceiling.
- **Behaviour metrics need interpretation.** "Long-term investment reduction near horizon end" is a good sign in this env but is context-dependent; a model that generalises this over-eagerly to short horizons may look worse elsewhere.
- **Seed sensitivity on 20-season runs is large.** Report multi-seed means and don't over-index on a single simulation run.

## Sources

- Paper: *FM-Bench: A Benchmark for Long-Horizon Management with Competing Agents* — Wang et al., 2026 — [arXiv 2608.18423](https://arxiv.org/abs/2608.18423) — introduces the simulator, both tracks, and the fifteen-model study.
