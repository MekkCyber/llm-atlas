# MATH-500
*Depth — the 500-problem evaluation subset of Hendrycks' MATH dataset.*

**TL;DR:** A curated **500-problem subset of the MATH test set**, introduced by OpenAI's *Let's Verify Step by Step* (Lightman et al. 2023) as a stratified sample from the 5,000-problem MATH test split after reserving the rest for PRM training. Covers all 7 MATH subjects and all 5 difficulty levels (level 1 easy → level 5 AIME-like). **LaTeX input, boxed-LaTeX output, programmatic grading**. The canonical "fast, hard, broad" math eval — smaller and quicker to run than full MATH, harder and broader than GSM8K. Essentially **saturated** at the reasoning frontier (95–98%); still used as a regression check and for comparing smaller models.

**Prereqs:** *(none)*
**Related:** [aime](aime.md) · [prm](../post-training/reasoning/prm.md) · [orm](../post-training/reasoning/orm.md)

---

## What it is

The MATH dataset (Hendrycks et al., NeurIPS 2021, arXiv 2103.03874):
- **12,500 problems** total: 7,500 train / 5,000 test.
- **7 subjects**: Prealgebra, Algebra, Number Theory, Counting & Probability, Geometry, Intermediate Algebra, Precalculus.
- **5 difficulty levels**, AoPS-style: Level 1 (easy) to Level 5 (AIME-class).
- **LaTeX** input and output. Final answer wrapped in `\boxed{...}`.

MATH-500 is the **500-problem subset** of the MATH test set chosen by Lightman et al. (OpenAI, "Let's Verify Step by Step", arXiv 2305.20050) as the held-out eval slice after the other 4,500 test problems were folded into PRM800K's training mix. The paper, Section 2.4 + Appendix C:

> *"we include data from 4.5K MATH test problems in the PRM800K training set, and we therefore evaluate our models only on the remaining 500 MATH test problems."*

The 500 were selected to be a representative uniformly-stratified sample across subjects and difficulty levels. The canonical JSONL lives in `openai/prm800k` (`prm800k/math_splits/test.jsonl`).

Everyone subsequent (R1, o1, Kimi k1.5, Gemini, Qwen3) uses the same 500-problem set — it's become the standard.

---

## How it works as an LLM eval

### Input / output

- **Input**: a LaTeX math problem.
- **Output**: LaTeX reasoning, with the final answer wrapped in `\boxed{...}`.
- **Grading**: programmatic equivalence on the boxed answer.

The grading function has three popular implementations:
- **Hendrycks grader** — the original from the MATH repo. Normalizes fractions, strips units, alphabetical variable ordering, `1/2 ≡ 0.5 ≡ \frac{1}{2}`.
- **Minerva grader** (Lewkowycz et al. 2022) — stricter / different normalization.
- **Lightman `math_equivalence` grader** — used in PRM800K, slightly different from the Hendrycks original.

Results can shift a few points between graders. When reproducing a number, check which grader the paper used.

### Metric

Exact-match-after-normalization on the boxed final answer. Reported as accuracy:
- **Pass@1** at `T = 0` or at low temperature.
- **Pass@1 averaged over N samples** for reasoning models (same convention as [aime](aime.md)).

### Why "500" and not "full MATH test"

- **Contamination management**: 4,500 of the 5,000 MATH test problems were used in PRM training data. Evaluating on them would contaminate comparisons against PRM-trained baselines.
- **Speed**: 500 × ~32k-token reasoning traces is ~16M tokens per evaluation. Full 5,000 is 10× that. Running MATH-500 costs a feasible ~$10–50 on a frontier API; full MATH would be expensive.
- **Coverage**: stratified sampling preserves subject / level mix, so the 500 is a representative slice.

---

## Why it matters

- **Canonical hard-math eval** post-2023. Everyone reports it. MATH-500 is the "did the reasoning training actually help" regression check.
- **Decent discriminator in the 50–95% range.** Below 50% is weak non-reasoning or small models; 50–85% is the current frontier of non-reasoning instruct; 85–98% is the reasoning-model regime. Clean separation across model classes.
- **Cheap to run.** 500 problems, no live judge, no LLM-as-a-judge — just a regex plus a normalization function. Reproducible.
- **Covers broad math.** Seven subjects, five levels. A model can't saturate MATH-500 by being good at one kind of math; it has to handle geometry, number theory, and precalculus in addition to algebra.

---

## Gotchas & tricks

- **Saturated at the frontier.** Current reasoning models hit 95–98%; discrimination above 95% is mostly noise. Use AIME 2024/2025 for harder discrimination. MATH-500 is still useful for smaller or non-reasoning models where headroom remains.
- **Contamination risk is high.** The MATH test set has been public since 2021. By now essentially every pretraining corpus has seen it. Scores are believed inflated, particularly on non-reasoning models that rely on memorization. The 500-problem subset doesn't fix this — it's contamination-suspect too.
- **Grader disagreement.** ~2–5 point swings are possible across Hendrycks / Minerva / Lightman graders. If a paper's number looks anomalous, check the grader.
- **Level mix matters.** Accuracy on level-1 is near-saturated for most modern models; level-5 is where the ceiling lives. Reporting overall accuracy without per-level breakdown obscures where a model actually struggles.
- **LaTeX output format is fragile.** A correct answer in the wrong LaTeX form can fail grading (e.g., `\tfrac{1}{2}` vs `\frac{1}{2}`). Normalization handles most cases but not all. Long tails of "grader-specific false negatives" are real.
- **No unit/proof problems.** The MATH dataset was built with the constraint that every answer is programmatically checkable. Proof-based and open-ended problems were excluded at collection time. This is good for grading, bad for pedagogical breadth — MATH-500 doesn't test a model's ability to write a proof.
- **Don't confuse with full MATH or GSM8K.** Full MATH (5,000 test) is rarely reported anymore; people report MATH-500 instead. GSM8K (1,319 test) is a different dataset (grade-school word problems) and is now too easy for the frontier.

---

## Typical modern numbers

| Model | MATH-500 Pass@1 |
|---|---|
| DeepSeek-R1 | 97.3% |
| Kimi k1.5 (long-CoT) | 96.2% |
| Kimi k1.5-short w/ rl | 94.6% |
| o1 (full) | 94.8% |
| o1-mini | 90.0% |
| QwQ-32B-Preview | 90.6% |
| DeepSeek-V3 | 90.2% |
| GPT-4o | 74.6% |
| Claude 3.5 Sonnet | 78.3% |
| Llama-3.1-405B-Instruct | 73.8% |
| 7B open reasoning (R1-Distill-Qwen-7B class) | 80–93% |

---

## Sources

- Paper: *Measuring Mathematical Problem Solving With the MATH Dataset* — Hendrycks et al., NeurIPS 2021 D&B, arXiv 2103.03874 — the MATH dataset.
- Paper: *Let's Verify Step by Step* — Lightman et al., OpenAI, 2023, arXiv 2305.20050 — introduces the MATH-500 subset (Section 2.4, Appendix C) and the canonical JSONL split.
- Repo: `openai/prm800k` — contains `prm800k/math_splits/test.jsonl`, the canonical MATH-500 file.
- Paper: *Minerva: Solving Quantitative Reasoning Problems with Language Models* — Lewkowycz et al., 2022, arXiv 2206.14858 — Minerva-style grader.
- Paper: *DeepSeek-R1* — DeepSeek-AI, 2025, arXiv 2501.12948 — reports the 97.3% headline.
- Paper: *Kimi k1.5: Scaling Reinforcement Learning with LLMs* — Moonshot AI, 2025, arXiv 2501.12599 — reports 96.2% long-CoT, 94.6% short-CoT.
