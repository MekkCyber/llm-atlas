# Context Pruning for Deep-Research Agents
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** For long-horizon research agents that iterate retrieval → aggregation → synthesis, **where you prune matters more than how you prune**. Pruning at the pre-retrieval stage — with cheap heuristics — cuts token usage by **up to 73%** with little quality loss; learned pruning is competitive on specific trade-offs but no single rule dominates on quality, efficiency, and faithfulness simultaneously. Established in *Not Worth Another Token* (UT Austin / Adobe Research / UMass Amherst, 2026).

**Prereqs:** [README.md](README.md)
**Related:** [../inference/README.md](../inference/README.md)

---

## What it is

A **deep research agent** solves an open-ended question by iterating: search → read → decide next query → aggregate → eventually synthesize a report. Context grows monotonically; the marginal value of each new piece of evidence declines quickly. Every token still costs money and dilutes the synthesis signal.

Pruning candidates:

- **Pre-retrieval.** Decide *which queries to run* or *which sources to hit* — cut candidate documents before they enter the pipeline.
- **Post-retrieval.** After retrieval, drop documents / chunks unlikely to help the final synthesis.
- **Pre-synthesis.** Right before writing the report, drop the least useful of what remains.

Two scoring families:

- **Lightweight heuristics.** BM25-style relevance, source authority, freshness, redundancy against what's already in context.
- **Learned value model.** A model trained to predict per-item marginal contribution to answer quality.

## How it works

The paper's central experiment: sweep the *stage* (pre-retrieval / post-retrieval / pre-synthesis) crossed with *scoring rule* (heuristic vs learned) across a benchmark of long-horizon research queries. Measure token cost, answer quality, and faithfulness.

**Central finding:** stage placement dominates scoring rule.

- **Pre-retrieval pruning** yields the largest end-to-end savings — it's the only place that reduces *retrieval* cost as well as later prefill.
- **Post-retrieval pruning** primarily reduces prefill cost of subsequent reasoning steps.
- **Pre-synthesis pruning** mainly refines the final report's context; it can't recover already-spent retrieval tokens.

Best operational recipe from the paper:

- Pre-retrieval **lightweight heuristic** (relevance + source authority + dedup).
- Optional post-retrieval learned value model when quality-critical.
- Skip pre-synthesis pruning unless the synthesis prompt is already at context limits.

Concrete number: this recipe reduces token usage by **up to 73%** with negligible answer-quality loss.

## Why it matters

- **Cost dominates deep-research agents.** These agents commonly burn 100K–1M tokens per query. Where you prune determines whether the cost curve is bearable.
- **Actionable, not clever.** Anyone shipping a research-agent product can implement pre-retrieval heuristic pruning in an afternoon.
- **Kills a common temptation.** Building a learned value model *for late-stage pruning* is a lot of engineering; the paper shows it's the wrong bet.

## Gotchas & tricks

- **Faithfulness is not the same as answer quality.** Aggressive pruning can preserve answer quality while dropping the sources needed to *cite* the answer. Track faithfulness separately.
- **Heuristic thresholds are query-dependent.** A single global threshold under-prunes easy queries and over-prunes hard ones; make the pruning budget proportional to expected effort.
- **Learned pruners overfit.** Value models trained on one agent's traces don't transfer cleanly to another agent stack; retrain or fall back to heuristics.
- **Retrieval already prunes.** If your retriever is already tight (small top-k), post-retrieval pruning has little left to do — the win moves to pre-retrieval query selection.

## Sources

- Paper: *Not Worth Another Token: Marginal Value Estimation for Efficient Deep Research Agents* — Kolukuluru, Ashok, Arora, Ciccarelli, Kumar, Nie, Dernoncourt, Basu, Rossi, Lipka (UT Austin / Adobe Research / UMass Amherst), arXiv 2608.08389, 2026.
