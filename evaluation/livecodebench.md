# LiveCodeBench
*Depth — the continuously-updated competitive-programming benchmark designed to be contamination-resistant.*

**TL;DR:** A code benchmark whose problems are **scraped weekly from LeetCode, AtCoder, and Codeforces** contests and tagged with release date, so you can evaluate on problems released *after* a model's training cutoff. Four scenarios: **code generation, self-repair, test output prediction, code execution**. Pass@1 via hidden tests. Jain et al. 2024 (arXiv 2403.07974). **Not saturated** — frontier reasoning models land 60–85%, non-reasoning frontier 35–55%. The standard "hard code" benchmark for reasoning LLMs as of 2025.

**Prereqs:** *(none)*
**Related:** [humaneval](humaneval.md) · [codeforces-benchmark](codeforces-benchmark.md) · [aime](aime.md)

---

## What it is

Paper: *LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code*, Jain et al., UC Berkeley + MIT + Cornell, arXiv 2403.07974 (v1 Mar 2024, v2 Jun 2024).

- **Source platforms** (Section 3.2):
  - **LeetCode** — all weekly contests.
  - **AtCoder** — Beginner Contests (ABC) only; ARC/AGC excluded.
  - **Codeforces** — Div. 3 and Div. 4 only.
- **Tagging**: each problem is labeled with its **contest release date `D`**.
- **Live update**: new contests are added on a rolling basis. Current release sizes:
  - **v5** (through Jan 2025): ~880 problems.
  - **v6** (through Apr 2025): ~1055 problems.
  (Counts vary slightly by source — livecodebench.github.io, the `livecodebench/code_generation_lite` HF dataset, and v5/v6 GitHub releases.)
- **Difficulty tiers**: LCB-Easy / LCB-Medium / LCB-Hard, derived from platform rating tags.
- **Average ~17 hidden tests per problem.**

"Live" solves the contamination problem by letting each paper report on a **time window** — e.g., "LiveCodeBench 2024-08 to 2025-01, 279 problems" — restricted to problems released *after* the model's training cutoff.

Repo: https://github.com/LiveCodeBench/LiveCodeBench. Site: https://livecodebench.github.io.

---

## Four evaluation scenarios

### 1. Code Generation

- **Input**: natural-language problem statement (like a LeetCode/AtCoder prompt).
- **Output**: a complete program.
- **Grading**: Pass@1 via hidden platform tests.
- **The headline scenario** — what people mean when they say "LiveCodeBench score."

### 2. Self-Repair

- **Input**: a (usually failing) program + a failing test case + error output.
- **Output**: a corrected program.
- **Grading**: Pass@1 on the repair against the full test set.
- Tests the model's ability to use feedback.

### 3. Code Execution

- **Input**: program + input.
- **Output**: predicted program output.
- **Grading**: `f(input) == predicted_output`, exact match.
- Inspired by CRUXEval; tests program-understanding without requiring generation.

### 4. Test Output Prediction

- **Input**: a natural-language problem + example input.
- **Output**: expected output for that input (no code).
- **Grading**: exact match on the predicted output.
- Tests reasoning about problem semantics, not code synthesis.

---

## How it works as an LLM eval

### Metric

- **Pass@1** — default, fraction of problems passing all hidden tests on first sample.
- **Pass@5 / Pass@10** — via Codex-style unbiased estimator (see [humaneval](humaneval.md)).

### Typical reporting

Modern papers report a time-window subset:
- R1 paper (Jan 2025): LiveCodeBench **2024-08 to 2025-01**, 279 problems, **65.9%** (R1), 53.1% (o1-mini), 33.4% (GPT-4o).
- Kimi k1.5 paper: reports two LCB numbers (long-CoT 62.5%, short-CoT 47.3%), time window unspecified in the body but window is tied to training cutoff.
- Earlier papers (pre-R1): often reported "LCB 05/23 – 03/24" or similar.

Always check the window — it's typically in the paper's table footnote or eval section.

### Platform test handling

- Where platform tests are available (LeetCode's hidden tests scraped at submission time), LCB uses them.
- Where not available, LCB uses **LLM-generated tests** (GPT-4-Turbo driven, per the paper). These can be weaker than true hidden tests, which is a known limitation.

---

## Why it matters

- **The only major code benchmark with a real contamination story.** Every other code benchmark (HumanEval, MBPP, APPS) has been in pretraining data for years. LCB's rolling time-stamped release makes clean post-cutoff evaluation possible.
- **Not saturated.** Frontier reasoning models top out ~85%; non-reasoning frontier ~55%; small open code models ~10–25%. Broad discrimination across the capability spectrum.
- **Tests multiple dimensions.** Generation + repair + execution + test prediction gives four orthogonal signals. A model that memorizes solutions (high generation) but can't reason about programs (low execution/test-prediction) is caught.
- **Contest-problem quality.** Problems come from curated competitive-programming contests, not crowd-sourced code Q&A. Harder, more precisely specified, better-tested than HumanEval-class problems.

---

## Gotchas & tricks

- **Contamination creep even in "live" problems.** The paper's Figure 1 shows clear contamination signatures for DeepSeek-Coder and GPT-4o on pre-cutoff contests. After contests are public, test problems may appear in Common Crawl / forums / blog solutions within weeks. "Live" is **relative to training cutoff**, not absolute contamination-free.
- **Time window choice is load-bearing.** A paper reporting LCB 2024-08 → 2025-01 (post-R1-cutoff) is honest; one reporting LCB all-time is almost certainly contaminated for training cutoffs after 2023. Compare like-for-like windows.
- **Test quality varies.** LeetCode problems with scraped hidden tests are well-tested. Problems with LLM-generated tests may be under-tested — correct solutions fail on edge cases the LLM didn't think of, or incorrect solutions pass because tests are too lenient.
- **Different subsets have different difficulty distributions.** LCB-Easy is near-saturated at the frontier; LCB-Hard is where discrimination lives. Aggregate numbers can hide this.
- **Codeforces subset is tiny.** In the original 511-problem release, only ~9 problems were Codeforces. Larger now but still a minority. LCB is primarily a **LeetCode + AtCoder** benchmark, not Codeforces — for Codeforces-style evaluation use [codeforces-benchmark](codeforces-benchmark.md).
- **Tests are fixed.** Unlike Codeforces Elo (recomputed per contest), LCB's tests don't change after release. Models that pattern-match test signatures can exploit them.
- **Competing v5/v6 versions.** Paper + website are sometimes out of sync on exact problem counts. Cite the version you used.
- **Not a good benchmark for code style or efficiency.** Only correctness is measured; all passing solutions score equal regardless of time/space complexity or readability. For style/quality eval, use separate benchmarks (e.g., CodeJudge).

---

## Typical modern numbers (LiveCodeBench code generation, Pass@1, post-2024 window)

| Model | LCB Pass@1 |
|---|---|
| DeepSeek-R1 | 65.9% |
| Kimi k1.5 (long-CoT) | 62.5% |
| Kimi k1.5-short w/ rl | 47.3% |
| o1 (full) | 67.2% |
| o1-mini | 53.1% |
| QwQ-32B-Preview | 40.6% |
| DeepSeek-V3 | 40.5% |
| Claude 3.5 Sonnet | 36.3% |
| GPT-4o | 33.4% |

Frontier reasoning: 60–85%. Non-reasoning frontier: 33–55%. 7–8B open models: 10–25%.

---

## Sources

- Paper: *LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code* — Jain, Han, Liu, Kazemi, Ishizaki, Hamann, Kim, Lee, Yuan, Stoica, Koyejo, Kelley, Sen, Yue, UC Berkeley + MIT + Cornell, 2024, arXiv 2403.07974.
- Site: https://livecodebench.github.io — leaderboard + version history.
- Repo: https://github.com/LiveCodeBench/LiveCodeBench.
- HF dataset: `livecodebench/code_generation_lite`.
- Paper: *DeepSeek-R1* — 2025, arXiv 2501.12948 — uses the 2024-08 to 2025-01 window.
- Paper: *Kimi k1.5: Scaling Reinforcement Learning with LLMs* — Moonshot AI, 2025, arXiv 2501.12599 — reports long-CoT and short-CoT LCB numbers.
- Related: *CRUXEval* — Gu et al. 2024 — the inspiration for the code-execution scenario.
