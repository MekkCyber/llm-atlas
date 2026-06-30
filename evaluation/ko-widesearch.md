# Ko-WideSearch
*Depth — a Korean breadth-search web-agent benchmark for exhaustive set enumeration.*

**TL;DR:** Most web-agent benchmarks evaluate *depth* — finding one obscure answer behind a chain of constraints. Ko-WideSearch evaluates **breadth**: tasks name a set-parent entity (a TV season, a dynasty, a league, an administrative region, an election) and require the agent to enumerate the full membership plus a per-item attribute table. Grading is via Item-, Column-, and Row-F1. 228 tables over 190 entities, 16 categories, three difficulty tiers controlled by two structural knobs.

**Prereqs:** [evaluation README](../evaluation/README.md)
**Related:** [agents README](../agents/README.md)

---

## What it is

A multilingual web-agent benchmark targeting *closure* (have I enumerated everything?) rather than *depth* (have I found the one answer?). Built specifically in Korean to expand non-English agent eval.

## How it works

**Task shape.** Given a set-parent entity (e.g. "members of the National Assembly of the 21st Republic of Korea"), the agent returns:

1. The full member set.
2. A per-member attribute table whose columns are specified by the task (party, term, district, …).

**Grading.**

- **Item-F1**: did you enumerate the right set?
- **Column-F1**: are the per-attribute columns populated correctly?
- **Row-F1**: per-row joint correctness.

**Difficulty knobs.** Two structural knobs are tuned independently to set tier difficulty:

- *Table width* (number of attribute columns).
- *2-D composite key* (e.g. "season × episode" instead of "season"), which exponentially expands membership.

Across tiers, cross-product membership climbs from **0% → 100%**, giving a continuous difficulty sweep.

**Synthesize-and-verify pipeline.** Automated construction with explicit verification — necessary because breadth gold sets rot quickly if any one cell is wrong or one item is missing.

## Why it matters

- **Web-agent benchmarks are biased toward needle-in-a-haystack.** Optimizing for them quietly biases agents away from completeness, deduplication, and per-item verification.
- **Non-English breadth coverage.** Korean-specific entities stress real cross-lingual web navigation; agents that succeed on English-only breadth often fall over on KR-only sources.
- **Tunable difficulty** makes Ko-WideSearch a good fine-grained measurement instrument, not a single-number leaderboard.

## Gotchas & tricks

- Composite-key tiers are far harder than width tiers at equal item count — don't conflate them.
- The synthesize-and-verify pipeline assumes a reliable source-of-truth corpus; benchmark integrity drifts as the underlying web does.
- F1 metrics depend on canonicalization (Item-F1 needs name normalization); the paper specifies a canonicalizer that the agent does not get to see during evaluation.

## Sources

- Paper: *Ko-WideSearch: A Korean Breadth-Search Benchmark for Exhaustive Set Enumeration by Web Agents* — arXiv:2606.27595 — https://arxiv.org/abs/2606.27595
