# Predictive Validity (for agent benchmarks)
*Depth — rank stability across in- and out-of-distribution evaluation as the headline metric for agent leaderboards.*

**TL;DR:** Position paper (IBM, 2026) arguing that aggregate-score leaderboards systematically underspecify deployed-agent quality. Rankings derived from in-sample mean scores **do not transfer** to out-of-distribution deployment, with direct evidence from public-to-hidden agent competition retrospectives. Proposes ranking by **predictive validity** — the correlation between in-sample and out-of-sample rank — and a twelve-tier measurement apparatus exposing deployment-relevant dimensions that single-leaderboard metrics collapse.

**Prereqs:** *(none)*
**Related:** *(none in graph yet)*

---

## What it is

Predictive validity, borrowed from psychometrics, asks: **does the in-sample score predict the out-of-sample rank?** For an agent benchmark, "in-sample" is the public eval set the benchmark was designed around; "out-of-sample" is any near-distribution deployment setting (new asset class, alternative orchestration, retrieval-strategy change, multimodal extension). A benchmark with high predictive validity rank-orders configurations the same way deployment does. A benchmark with low validity rank-orders them differently — i.e., its leaderboard is misleading.

## How it works

The paper operationalizes the framework with:

- **Three falsifiable OOD criteria.** Explicit thresholds for what counts as "the rankings transferred." Pre-registered against future evaluation runs.
- **Fourteen parallel deep-dive studies on a single MCP-based industrial benchmark.** New asset classes, alternative orchestrations, retrieval strategies, reasoning modes, infrastructure optimizations, evaluation-methodology probes. Consolidated with seven prior benchmarks for the cross-benchmark analysis.
- **Twelve-tier measurement apparatus.** Each tier captures a deployment dimension (latency under load, tool-failure handling, schema drift, policy drift, …) that single-scalar leaderboards collapse.
- **Empirical evidence base.** Public-to-hidden competition retrospectives provide direct rank-instability data — winners of the public phase routinely lose the hidden phase.

The output is not a new benchmark — it's a *meta-evaluation* framework that existing benchmarks (HELM, agent leaderboards) can be scored against.

## Why it matters

- Agent evaluation is at the crisis point image classification benchmarks hit in 2019: leaderboard saturation hides distribution-specific overfitting.
- "Predictive validity over aggregate score" reframes what benchmark authors *should* report. The framework is concrete enough to be adopted — pre-registered protocols, explicit thresholds.
- For model selection in production: this paper is the argument for **not** picking the model that tops the public leaderboard.

## Gotchas & tricks

- Predictive validity requires *paired* in-/out-of-sample evaluations, which most benchmark authors don't publish. The framework's value depends on the community shipping that data.
- It's a position paper plus measurement framework; existing evidence partly supports the claim but is admittedly thin. Treat the thresholds as proposals, not consensus.
- The twelve-tier apparatus is specific to MCP-based agent benchmarks — generalizing the tiers to non-tool-using LLM evals is non-trivial.
- Distinct from *contamination resistance* (which LCB-style live release addresses). Predictive validity is about *distribution shift*, contamination is about *data leakage* — orthogonal failure modes.
- Sister concept: external validity in social-science methodology. Reading that literature first helps.

## Sources

- Paper: *Beyond Static Leaderboards: Predictive Validity for the Evaluation of LLM Agents* — Patel et al. (~60 co-authors), IBM, 2026 — arXiv 2606.19704.
- Background: classical psychometrics — Cronbach & Meehl, *Construct Validity in Psychological Tests*, 1955.
- Related: HELM (Liang et al., 2022) — earlier multi-axis benchmarking effort the paper builds on.
