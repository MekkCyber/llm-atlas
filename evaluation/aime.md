# AIME — American Invitational Mathematics Examination
*Depth — the olympiad-qualifier math contest that became the canonical hard-math eval for reasoning LLMs.*

**TL;DR:** 15 problems per exam, 3 hours, no calculator, **integer answers 000–999**. Two sittings per year (**AIME I + AIME II = 30 problems**). The MAA runs it for top US high-school students as the gate between AMC 10/12 and the USAMO. AI researchers repurpose each year's 30 problems as an eval set: **AIME 2024**, **AIME 2025**, etc. Since answers are integers, grading is trivial exact-match — which is the whole point. Current frontier reasoning models (o1, R1, Kimi k1.5, Gemini 2.5 Pro) hit 77–92% on AIME 2024. Not a dataset with its own paper; a live human competition the AI community scrapes.

**Prereqs:** *(none)*
**Related:** [math500](math500.md) · [livecodebench](livecodebench.md) · [long-cot-rl](../post-training/reasoning/long-cot-rl.md)

---

## What it is

A real math competition:

- **Administered by**: Mathematical Association of America (MAA AMC series).
- **Who takes it**: high-school students who scored well on AMC 10/12.
- **Purpose**: qualifier for USAMO / USAJMO (the US olympiad team selection).
- **Format** (per AoPS wiki):
  - 15 problems per exam.
  - 3 hours, no calculator.
  - **Integer answers in `[0, 999]`**, written as three digits with leading zeros.
  - No partial credit; score is the count correct out of 15.
- **Two sittings per year**:
  - **AIME I** — early February.
  - **AIME II** — late February, ~2 weeks later. For students who missed AIME I or qualified via AMC 12A/B.
- "AIME `YYYY`" as an eval set typically means **30 problems = AIME I (15) + AIME II (15)**. Some papers use only AIME I (15 problems) — always check the table footnote.

It is **not a dataset with an accompanying paper**. Problems are released by the MAA, discussed on Art of Problem Solving (aops.com), and mirrored on Hugging Face (`Maxwell-Jia/AIME_2024`, `HuggingFaceH4/aime_2024`). MathArena curates its own versions with solutions.

---

## How it works as an LLM eval

### Input / output

- **Input**: a natural-language math problem, often with LaTeX notation.
- **Output**: a single integer in `[0, 999]`.
- **Grading**: exact match on the final integer, extracted from the response (typically a `\boxed{...}` or the last integer in the text).

The integer-answer format is what makes AIME an attractive RL verifier target — the reward function is one regex. No LLM judge needed, no ambiguity about equivalent algebraic forms.

### Metric conventions

- **Pass@1** — naive single-sample accuracy. For reasoning models at nonzero temperature, noisy on 30 problems.
- **"Pass@1 averaged over N samples"** — what R1 and Kimi k1.5 actually report. Operationally: sample `N` independent completions per problem at nonzero temperature, compute fraction correct per problem, average across problems. **This is not the unbiased Codex-style Pass@k estimator** — it's just mean accuracy across `N` samples. `N = 8, 16, 32, 64` in the wild.
- **Cons@N** / **maj@N** — majority vote over `N` samples per problem, then grade the aggregated answer. Typically 5–15 points higher than Pass@1 for reasoning models. OpenAI's o1 blog reports both: pass@1 74.4%, cons@64 83% on AIME 2024.

The variance of "pass@1 at T=0.6" on AIME 2024 is large enough — 30-problem set, a single problem swing is 3.3 points — that single-sample numbers are essentially unreportable. Averaged-over-N is the norm.

### Decoding settings for reasoning models

Standard DeepSeek-R1 setting (reproducible target): **temperature 0.6, top-p 0.95, max generation length 32768**. Long context is required — R1-class models routinely emit 4k–20k-token CoTs on AIME problems.

---

## Why it matters

- **Hard enough to matter.** Top reasoning models don't saturate AIME 2024 (77–92%); frontier-of-reasoning headroom is still visible. Compare MATH-500, which is saturated at ~96%+.
- **Unhackable grading.** Integer answers mean the verifier is `answer == reference`. No LLM judge, no scoring subjectivity. This is why reasoning-RL papers preferentially use AIME — the reward signal is clean.
- **Contamination-mitigable via AIME `YYYY+1`.** Every year the MAA releases a fresh 30-problem set. Using AIME 2025 in papers with training cutoffs before Feb 2025 is a genuine contamination guard. This is why every reasoning paper in 2025 reports both AIME 2024 and AIME 2025.
- **The reasoning benchmark leaderboards run on.** o1, R1, Kimi k1.5, Gemini 2.5 Pro, Claude 3.7 Sonnet with extended thinking, Qwen3 — all report AIME 2024/2025 as the headline math number.

---

## Gotchas & tricks

- **Small N (30 problems) → huge variance.** One problem ≈ 3.3 pp. A single "averaged over 8 samples" number can swing ±5 pp from run to run. Paper comparisons within 3 points are mostly noise.
- **Contamination in AIME 2024 is real and substantial.** The "Challenging the Boundaries of Reasoning" study (arXiv 2503.21380, Mar 2025) found statistically significant contamination signals for most 2024-era models on AIME 2024 — scores likely inflated 10–20 pts vs uncontaminated contests. Use AIME 2025 or other fresh olympiads for honest comparisons to 2024-trained models.
- **AIME I only vs full 30 problems.** Some earlier papers (pre-R1) report on AIME I only — 15 problems, even higher variance. Modern convention is both sittings (30 problems). Always confirm.
- **"Pass@1 avg over 8/16/32/64" vs "true Pass@1".** The averaged metric is what reasoning papers report. A "Pass@1" number without a sample count is underspecified — assume N=1 at temperature 0 unless stated otherwise.
- **Integer answers mean partial correctness gets no credit.** A right reasoning chain with an arithmetic slip gets zero. This is cleaner than MATH-500's LaTeX grader (no equivalence ambiguities) but can penalize models that reason correctly but miscount.
- **Some problems admit guessing tricks.** Integer-in-[0,999] has 1000 possible answers; guessing uniformly is ~0.1% expected pass rate. But "reasonable" integer answers (small positive integers, multiples of 10, etc.) are a much smaller set; a weak model emitting random guesses can score nonzero. Normalize mentally.
- **Problem 15 is usually much harder than Problem 1.** AIME problems are ordered by difficulty. Aggregate accuracy hides the ceiling — report problem-level accuracy distribution if you care about the reasoning frontier.
- **"AIME 2024" attribution varies.** Some papers include only the published problems; others include the test-day errata. Reproducibility requires publishing the exact problem set used.

---

## Typical modern numbers (AIME 2024, pass@1 averaged over N samples)

| Model | AIME 2024 |
|---|---|
| o1 (full) | 74.4% (cons@64: 83%) |
| o1-mini | 63.6% |
| DeepSeek-R1 | 79.8% |
| Kimi k1.5 (long-CoT) | 77.5% |
| Kimi k1.5-short w/ rl | 60.8% (short-CoT, ~3,272 tokens avg) |
| QwQ-32B-Preview | 50.0% |
| Claude 3.5 Sonnet | ~16% |
| GPT-4o | ~9–13% |

Non-reasoning frontier models cluster at 9–20%. Reasoning models range from 50% (Qwen's QwQ-32B-Preview) to ~90% (o3, frontier).

---

## Sources

- Official problem archive: https://artofproblemsolving.com/wiki/index.php/AIME_Problems_and_Solutions (AoPS wiki).
- MAA AMC program: https://www.maa.org/math-competitions (format, eligibility).
- HuggingFace mirrors: `Maxwell-Jia/AIME_2024`, `HuggingFaceH4/aime_2024`.
- MathArena leaderboard: https://matharena.ai/ (curated versions + model scores).
- Paper: *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning* — DeepSeek-AI, 2025, arXiv 2501.12948 — reports the "pass@1 averaged over 64 samples, temperature 0.6, top-p 0.95" convention.
- Paper: *Challenging the Boundaries of Reasoning* — 2025, arXiv 2503.21380 — contamination analysis on AIME 2024.
- Blog: *Learning to Reason with LLMs* — OpenAI, Sept 2024 — o1's AIME 2024 numbers and the pass@1 / cons@64 convention.
