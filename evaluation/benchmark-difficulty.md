# Benchmark Difficulty (and BenchFill)

*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A benchmark induces an empirical probability measure over *difficulty*. Two failure modes: **measure mismatch** (the empirical difficulty distribution doesn't match the benchmark's stated intent) and **coverage gaps** (whole difficulty bands have no items, so progress in that band is invisible). BenchFill is the fix: a targeted item-rewriting method that patches coverage gaps while preserving the benchmark's item style. The authors audit 25 benchmarks across 7 domains and find pervasive mismatch and gaps.

**Prereqs:** [mmlu](mmlu.md), [humaneval](humaneval.md)
**Related:** [../data/decontamination](../data/decontamination.md)

---

## What it is

A statistical framing of benchmark design. Each item has a latent difficulty (operationalized as e.g. average frontier-model failure rate, or a 2PL IRT $b$-parameter). The benchmark as a whole is a *sample* of (item, difficulty) pairs — equivalently, an empirical measure $\hat\mu_B$ over difficulty.

Two defects of $\hat\mu_B$:

- **Measure mismatch.** The benchmark claims to span, say, "high-school-to-olympiad math uniformly" but $\hat\mu_B$ is concentrated at medium difficulty — so a model that improves on olympiad-level questions moves the average almost not at all.
- **Coverage gap.** A difficulty band has zero items. A model that newly solves that band is undetectable.

## How it works

### Diagnostic metrics

For a benchmark $B$ with intended difficulty measure $\mu^*$:

- **Mismatch:** total-variation (or KL) distance between $\hat\mu_B$ and $\mu^*$.
- **Coverage:** maximum-length empty interval in the support of $\hat\mu_B$ relative to $\mu^*$.

Both are computable from standard eval logs if a difficulty estimate per item is available (frontier-model pass rate, IRT fit, human-grader estimate).

### BenchFill

To patch coverage gaps, select existing items whose difficulty is adjacent to a gap, then use an LLM to *rewrite* them at the needed difficulty while preserving domain, style, and answer format. The rewriting prompt is seeded with (a) the target difficulty estimate, (b) adjacent same-style items as style anchors, and (c) a verification pass to confirm the new item is well-formed. BenchFill is explicitly a retrofit — it doesn't generate items from scratch, so it inherits the benchmark's provenance and avoids contamination risk from entirely new material.

## Why it matters

- Benchmarks are the only training signal that persists across model releases. A benchmark that can't resolve where progress is happening is actively misleading leaderboards.
- Audit of 25 widely-used benchmarks (MMLU, GSM8K, HumanEval etc.) reveals that most are heavily skewed in difficulty — a general critique of how saturated vs un-saturated scores are read.
- Gives benchmark maintainers a concrete, cheap tool (BenchFill) to extend their benchmark's useful range without starting over.

## Gotchas & tricks

- Difficulty must be estimated somehow. Using frontier-model pass rate biases the measure toward what current models find hard, which drifts over time. IRT fits are more stable but need many models' scores per item.
- BenchFill-rewritten items can inherit biases of the LLM used to rewrite; the paper runs a human-verification pass on the rewrites.
- "Fixing" a benchmark changes its score distribution; publish both the original and the patched version to preserve historical comparability.
- Coverage gaps are not always bugs — a benchmark that focuses on medium-hard items may be *intentional*. Match the diagnostic to the benchmark's stated goal.

## Sources

- Paper: *Your Benchmark Is an Empirical Measure Over Difficulty* — Yu, Gong, Shen, Zhang, UIUC / UCSD / NYU, 2026 — [arXiv:2610.29927](https://arxiv.org/abs/2610.29927).
- Related: Item Response Theory background — Lord, 1980 — the 2PL IRT framework behind difficulty estimation.
