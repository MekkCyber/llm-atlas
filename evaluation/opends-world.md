# OpenDS-World
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A live-Web deep-search benchmark for LLM agents: 200 human-verified tasks that require multi-step planning, browser interaction, evidence synthesis, and failure recovery against the *actual* public internet, not a static snapshot. 12 frontier models evaluated. Best model: 39.0 % accuracy; human solvers: 21.1 % — unusual in that frontier models already clear the human baseline on a hard eval (HKUST Guangzhou + consortium, 2026).

**Prereqs:** [README.md](README.md)
**Related:** [../agents/README.md](../agents/README.md), [livecodebench.md](livecodebench.md)

---

## What it is

An agent benchmark that measures deep-search competence under realistic conditions: live Web, no cached corpus, no sanitised search API. Each task has a human-verified answer that requires an agent to form queries, open and read pages, follow links, and reconcile conflicting evidence. The eval records trajectory properties — tool usage, search cost, termination behavior, failure recovery — not just final-answer accuracy.

## How it works

- **Task shape.** Each of 200 tasks is a natural-language question whose answer is obtainable only by browsing: multi-hop facts, aggregations across pages, verification across sources, time-sensitive lookups. The verified answer is a short string or small structured object.
- **Environment.** Agents are given a browser tool and a search tool. The browser hits the real Web; no snapshot. The search tool returns actual engine results, not a curated index.
- **Scoring.** Primary metric: exact-match accuracy against human-verified answers. Secondary metrics: tokens of search output consumed, number of pages opened, time to termination, and recovery count (how often the agent recovers from a 404 / mis-rendered page / contradictory source).
- **Human baseline.** Expert human solvers given the same tooling and a time budget; the baseline (21.1 %) is intentionally hard — it uses tasks designed to resist pure recall.

## Why it matters

- **Live-Web first.** Previous agent benchmarks froze the environment (fixed Wikipedia snapshot, canned search results); real deployments drift continuously. OpenDS makes drift a feature, not a confound.
- **Trajectory-aware.** Evaluating only on final-answer accuracy encourages brute-force browsing. OpenDS also surfaces inefficient and non-recovering trajectories.
- **Frontier > human.** For the first time on a public live-Web deep-search eval, best model > best human, which reframes "what deep-search agents are for." Expect this to replace static deep-search evals as the primary leaderboard.

## Gotchas & tricks

- **Drift makes score comparability loose.** Running the same agent six months apart may yield different scores because the Web changed. The paper notes this and recommends dated runs.
- **Rate limits.** Honest live-Web evaluation hits CAPTCHAs, bans, and session limits. Reproducing the paper's harness requires patience and a clean IP pool.
- **Human baseline is a floor, not a ceiling.** 21.1 % reflects the task set's difficulty for humans without specialised retrieval tools; it does not mean humans are "beaten" at research in general.
- **No partial credit.** Exact-match hides improvement; the paper's secondary metrics are what actually distinguish close systems.

## Sources

- Paper: *Deep Search in Real-World Web Environments* — Yu et al., HKUST (Guangzhou) + consortium, 2026 — [arXiv:2610.30181](https://arxiv.org/abs/2610.30181).
