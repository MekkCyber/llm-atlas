# PRISM

*Depth — a multi-dimensional benchmark for evaluating LLM peer reviewers against human reviewers.*

**TL;DR:** Most LLM-reviewer evaluations rely on surface metrics (ROUGE, BLEU) or unconstrained LLM-as-a-judge prompting that conflates fluency with rigor. PRISM grounds each review-quality dimension in a verifiable procedure — argument mining for depth-of-analysis, retrieval-augmented verification for novelty, consensus-based scoring for prioritization — over a stratified corpus of real ICLR / ICML / NeurIPS reviews. Reveals that LLM reviewers can match or beat humans on *individual* dimensions but none matches the balanced human profile across all four.

**Prereqs:** [README.md](README.md)
**Related:** [../agents/README.md](../agents/README.md)

---

## What it is

A benchmarking framework for automated peer reviewers (LLM-based or hybrid) that measures four review qualities independently rather than collapsing them into a single judge-LLM rating. Targets the trap where a fluent but shallow review scores high under aggregate metrics.

## How it works

Four scoring dimensions, each with a structured procedure:

1. **Depth of Analysis** — argument mining over the review extracts claims and supporting evidence; scores depth by the chain length and the specificity of evidence.
2. **Novelty Assessment** — retrieval-augmented verification cross-checks novelty claims in the review against the literature index.
3. **Flaw Identification & Major Issues Prioritization** — consensus-based scoring against held-out human "major issue" labels; reviewers are credited for catching the issues human reviewers also caught.
4. **Multi-dimensional Constructiveness** — measures whether the review provides actionable suggestions, not just complaints.

The benchmark corpus is a stratified sample of reviews from ICLR, ICML, and NeurIPS, covering high-, medium-, and low-quality papers. Five leading automated reviewer systems and human reviewers are scored on the same instances.

## Why it matters

- Replaces surface-level review evaluation (which has driven misleading "LLM matches human reviewer" claims) with grounded per-dimension scoring.
- Reveals *specialization profiles*: each LLM reviewer system has a distinct blind spot — strong on one dimension, weak on another. Aggregate metrics hide this. Implication: deploy LLM reviewers as targeted supplements (novelty checker, completeness auditor), not standalone replacements.
- Provides a template for building dimensional benchmarks elsewhere — grounding each dimension in a verifiable procedure makes the evaluation itself less judge-LLM-biased.

## Gotchas & tricks

- The corpus is conference-specific. Generalization to journal reviews, workshop reviews, or domain-specific venues is not tested.
- "Novelty verification" requires a literature index — the score quality depends on the index's coverage. Stale indices systematically under-score novelty.
- Consensus-based scoring against human reviewers inherits any biases in the human reviewer pool (e.g., systematic blind spots for certain subfields).
- LLM reviewers can game depth-of-analysis by generating longer chains of trivial claims. The argument-mining scoring weights claim *specificity*; verify per-system that the depth score is not just length.
- Useful as a public leaderboard but also as a *training signal* — the per-dimension scores can serve as rewards for fine-tuning LLM reviewers toward balance rather than specialization.

## Sources

- Paper: *PRISM: A Multi-Dimensional Benchmark for Evaluating LLM Peer Reviewers* — Loc, Viet, Khanh, Nguyen, Pham, Nguyen, Chawla, Buntine, Wong, Doan, Nguyen — VinUniversity / Notre Dame / Monash / UIUC, 2026 — [arXiv 2605.26730](https://arxiv.org/abs/2605.26730). Project: https://prism-benchmark.github.io/
