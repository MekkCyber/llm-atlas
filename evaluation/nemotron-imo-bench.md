# Nemotron-IMO-Bench
*Depth — one specific technique / evaluation, grounded in its source paper(s).*

**TL;DR:** A 200-problem olympiad-level math benchmark released with the Nemotron IMO-gold paper. Each problem is novel (not sourced from public IMOs / national olympiads) and requires a natural-language proof — not just an answer. Designed to evaluate proof-generation reasoning at IMO-hard difficulty without the public-web contamination that plagues existing math benchmarks.

**Prereqs:** [math500](math500.md), [aime](aime.md)
**Related:** [../post-training/reasoning/generate-verify-refine](../post-training/reasoning/generate-verify-refine.md), [../post-training/reasoning/long-cot-rl](../post-training/reasoning/long-cot-rl.md)

---

## What it is

Public olympiad benchmarks (past IMO/USAMO/IMO shortlists, AoPS) leak everywhere: solutions are online, models see them at pretraining, evaluations get inflated. Nemotron-IMO-Bench collects 200 **novel** olympiad-level problems written for this benchmark, at difficulty comparable to IMO shortlist entries.

Each item requires a full natural-language proof — a numeric-answer or multiple-choice format cannot capture olympiad-level reasoning. Grading is proof-based, following IMO-style partial-credit rubrics.

## How it works

- **Problem sourcing.** 200 fresh problems authored / curated for the benchmark, covering the usual IMO areas (algebra, combinatorics, geometry, number theory).
- **Proof-based evaluation.** Every problem asks for a proof, not just a final numeric answer. Grading follows an IMO-style partial-credit rubric.
- **Grading harness.** A separate grader — human, LLM-judged, or hybrid — evaluates the proof against the reference solution and rubric. The paper specifies which mode was used for each headline number.
- **Difficulty calibration.** Problem difficulty was calibrated against IMO shortlist entries so scores are comparable in spirit to IMO-style evaluation.
- **Released alongside the recipe.** Problems, references, rubrics, and grader code all released so the benchmark is reproducible without the paper's team.

## Why it matters

Existing math benchmarks (MATH, GSM8K, AIME) either don't reach olympiad difficulty or are heavily contaminated. Novelty + proof-based grading gives a cleaner signal for what a model *actually* knows about olympiad reasoning — and it's a natural test bed for future generate-verify-refine, tree-search, and long-CoT-RL work.

## Gotchas & tricks

- **Not the IMO itself.** Nemotron-IMO-Bench is a *proxy* — designed to correlate with IMO performance without being IMO problems. The paper's headline IMO number (30/42) is on the real IMO 2026; the 200-problem bench is separate.
- **Grader choice materially moves numbers.** LLM-judged and human-graded scores diverge by several points on hard problems. Report the grading mode alongside the number.
- **Novelty needs maintenance.** As models scrape the paper's release, the problems become contaminated. Plan a rotation for the benchmark to stay useful past release + 6 months.
- **Proof-based grading is expensive.** Human grading of 200 olympiad proofs is not cheap; LLM-judged grading is faster but sensitive to prompt and judge.
- **Partial credit matters.** An all-or-nothing grader wastes signal — most models get partial-credit on many problems, and the rubric captures that.

## Sources

- Paper: *An Open Recipe for IMO Gold: Training Nemotron for Olympiad Mathematics* — Moshkov, Ge, Armstrong, Du, Mahdavi, Gitman (NVIDIA), 2026 — https://arxiv.org/abs/2609.10712
- Case study: [nemotron-imo](../case-studies/nemotron-imo.md).
