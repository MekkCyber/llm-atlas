# Diverse Skill Routing (DSR)
*Depth — a diversity-aware reranker for LLM-agent skill retrieval that treats routing as complementary-set selection instead of independent Top-K ranking.*

**TL;DR:** LLM agents routinely rely on external skills or tools; when there are many, a router has to pick a small subset per request. Pointwise Top-K rankers score each candidate against the query independently — which wastes context budget on functionally redundant skills. **DSR** replaces Top-K with a **Determinantal Point Process** over a *query-residual diversity kernel* that penalizes redundant skill overlap while suppressing the penalty caused only by shared query relevance. Improves recall and full coverage over strong pointwise rerankers on SkillRouter, with larger gains on multi-skill queries.

**Prereqs:** *(none in-graph yet)*
**Related:** [skill-optimization](skill-optimization.md)

---

## What it is

Skill routers sit between a user query and a large registry of skills / tools. The router picks the top few skills to expose to the agent's context, since exposing all skills would blow the context budget or dilute attention. Traditional routers:

1. Score each candidate skill against the query (relevance).
2. Sort by score.
3. Keep the top $K$.

The failure mode: functionally *redundant* skills all score high on the query, so Top-K fills its budget with duplicates. Complex tasks that need **complementary** skill sets — e.g. "find a paper AND summarize it AND format the citation" — get worse coverage than they'd get from a diverse-but-slightly-less-relevant subset.

DSR reframes routing as **subset selection** rather than pointwise ranking.

## How it works

### Determinantal Point Processes for skill selection

A **DPP** models the probability of selecting a subset as proportional to the determinant of a kernel submatrix. Determinants reward diversity: highly similar items increase collinearity, which shrinks the determinant, which lowers the subset probability. Formally, for a set $S$ of candidates and kernel $L$:

$$
P(S) \propto \det(L_S)
$$

Selecting the max-determinant subset of size $K$ gives a diverse-yet-relevant set.

### The query-residual diversity kernel

Standard DPP kernels combine relevance and similarity, but for skill routing they over-penalize skills that are similar to each other just because they're **both relevant to the query**. DSR builds a **query-residual** kernel: after conditioning on the query, redundancy is measured on the *residual* skill embedding — i.e. what's left after subtracting the query-relevance component.

The effect: two skills that both help the query but are functionally different survive together; two skills that overlap in *how* they help are collapsed.

### Reranking stage

DSR is a **reranking layer** on top of an existing pointwise scorer, not a from-scratch retrieval system. Take the top few dozen candidates from the pointwise scorer, then apply DSR's DPP-based selection to pick the final $K$. This keeps latency reasonable — DPP inference is $O(N^3)$ in the candidate set size, so the reranker's input has to be small.

## Why it matters

- **Multi-skill queries were the visible failure mode.** Pointwise Top-K is fine on single-skill queries; it degrades sharply when the request needs complementary skills. DSR's gains are largest exactly there.
- **Complementary-set framing is the right abstraction.** Skill routing has more in common with subset selection (submodular optimization, DPP) than with document retrieval; DSR is the first paper to lean fully into that framing for agent skill registries.
- **Reranker slot is cheap.** Slotting DSR on top of an existing pointwise scorer keeps the retrieval pipeline unchanged and gets diversity for free.

## Gotchas & tricks

- **Kernel calibration matters.** If the query-residual is under-weighted, DSR collapses back to a pointwise ranker; if over-weighted, it prefers diversity over relevance and gets *less* useful skills.
- **Candidate pool size caps DPP inference.** Reranking 20–50 candidates is fine; reranking thousands needs approximate DPP inference.
- **Not a substitute for a good pointwise scorer.** DSR only reorders what the scorer surfaces — a skill missing from the top-N pool cannot be recovered.
- **Metric matters.** Full-coverage (did every needed skill get retrieved?) shows DSR's win more clearly than raw recall.

## Sources

- Paper: *Beyond Top-k Skill Retrieval: Diversity-Aware Skill Routing for LLM Agents* — Wei, Yang, Basu, Chen, Zhao, Tu, Hu, Dernoncourt, Rossi, Eldardiry (Virginia Tech / Adobe / USC), 2026 — [arXiv:2609.05824](https://arxiv.org/abs/2609.05824).
- Background: *k-DPPs: Fixed-Size Determinantal Point Processes* — Kulesza & Taskar, 2011 — the DPP variant DSR builds on.
