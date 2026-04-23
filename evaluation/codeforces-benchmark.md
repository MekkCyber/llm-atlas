# Codeforces as an LLM Benchmark
*Depth — how to operationalize the competitive-programming platform Codeforces as a model eval.*

**TL;DR:** Codeforces is a real competitive-programming platform that hosts contests weekly and assigns Elo ratings to ~1M registered human contestants. AI papers evaluate models by **simulating participation in recent Codeforces contests**, computing an Elo rating and percentile against the human population. Not a static benchmark — **a methodology**. OpenAI's o1 reported **Elo 1673, 89th percentile** via this approach. The **CodeElo** framework (Qwen, 2025, arXiv 2501.01257) standardizes the methodology for reproducibility by submitting directly to Codeforces servers.

**Prereqs:** *(none)*
**Related:** [livecodebench](livecodebench.md) · [humaneval](humaneval.md)

---

## What it is

**Codeforces** (https://codeforces.com) is a competitive-programming platform:
- Weekly public contests with 4–8 problems, 2–3 hours duration.
- Elo-style rating (specifically a Glicko-like system) updated after each contest.
- ~1M registered active users; thousands participate in each contest.
- Problems are divided into **divisions** (Div. 1 for strongest contestants, Div. 4 for weakest), with overlapping content.

As an LLM benchmark, "Codeforces" means: give a model recent Codeforces contest problems under contest conditions, submit its solutions, compute the rating the model would have received had it participated as a contestant, convert to **percentile** against the human distribution.

This is **not a paper benchmark**. Papers describe the methodology; different papers use different details. Compared to static benchmarks like [livecodebench](livecodebench.md) or [humaneval](humaneval.md), the output is a single number with intuitive meaning — "this model would rank in the top 11% of Codeforces contestants."

---

## How it works as an LLM eval

### General methodology

1. **Pick a set of contests** — typically recent (post-training-cutoff) to avoid contamination.
2. **For each contest**: give the model each problem, ask for a solution, submit for judging.
3. **Evaluate as if the model had participated** — at the time of the contest, against that contest's other submissions, under the same scoring system.
4. **Aggregate** across contests into an Elo rating and convert to percentile.

### OpenAI's o1 (2024) specifics

From the o1 blog (*Learning to Reason with LLMs*, September 2024):
- Evaluated on "simulated Codeforces contests."
- Reported: **o1-preview 1258 rating**, **o1-mini 1650**, **o1 (full) 1673 / 89th percentile**.
- Exact methodology (how many contests, how many submissions per problem, time limits, sampling) is **not fully disclosed** in the public blog. The later OpenAI *Competitive Programming with Large Reasoning Models* paper (arXiv 2502.06807) gives more detail for o3.

### CodeElo (2025)

Paper: *CodeElo: Benchmarking Competition-level Code Generation of LLMs with Human-comparable Elo Ratings*, QwenLM, Jan 2025, arXiv 2501.01257.

- Collects ~6 months of recent Codeforces contest problems with metadata (division, difficulty rating, algorithm tags).
- **Submits solutions directly to the Codeforces platform** for official judging (correct handling of special judges, hidden tests, execution environment).
- Computes Elo via a standardized pipeline (fixed submission ordering, contest selection) — aligned with the real Codeforces rating system but with lower variance.
- Reports Elo + percentile for 30 open-source + 3 proprietary models.
- Reported: **o1-mini 1578, QwQ-32B-Preview 1261**, etc.
- Repo: https://github.com/QwenLM/CodeElo.

CodeElo is the closest thing to a reproducible standard. Papers that report a Codeforces percentile without citing CodeElo or an equivalent framework are using their own in-house methodology — numbers are not directly comparable across papers.

### Typical paper reporting

Papers report some combination of:
- **Elo rating** (e.g., 1673).
- **Percentile** against the human population (e.g., 89th).
- **Division** (Div. 1 / Div. 2 / Div. 3 / Div. 4) as a coarse tier.

The percentile-to-Elo mapping is roughly (approximate thresholds):
- Pupil ~1200–1399 → ~top 75%.
- Specialist ~1400–1599 → ~top 50%.
- Expert ~1600–1899 → ~top 25%.
- Candidate Master ~1900–2099 → ~top 10%.
- Master ~2100–2299 → ~top 5%.
- International Master / Grandmaster ~2300+ → ~top 1%.

---

## Why it matters

- **Human-comparable metric.** Unlike raw accuracy, "89th percentile against 100k human contestants" gives immediate intuition for how strong a model is at competitive programming.
- **Very hard problems.** Codeforces Div. 1 problems are beyond most software engineers and most CS grads. The ceiling for LLM progress is high.
- **Contamination-guardable.** Using only post-cutoff contests gives clean signal.
- **Captures algorithmic reasoning.** Codeforces problems are mostly about algorithms and data structures, where pattern-matching from pretraining data has limits. Genuine reasoning pays.
- **The headline "our model can out-compete humans" benchmark.** O1's 89th percentile and subsequent claims by o3, Gemini 2.5, R1 have made Codeforces the go-to comparison for frontier reasoning.

---

## Gotchas & tricks

- **Elo is sensitive to methodology.** The "When Elo Lies" literature documents how submission ordering, contest selection, number of tries per problem, and time budgets all change the Elo by ±100–200 points. Numbers from different papers are not directly comparable unless the methodology matches.
- **`k` submissions per problem matters.** Humans get ~5 submissions with wrong-answer penalties. Some LLM evaluations allow unlimited submissions, best-of-k, or majority voting — inflating effective Elo above what a human contestant could achieve in the same time.
- **Time limits are often ignored.** Humans have 2–3 hours per contest. LLM evaluations rarely enforce wall-clock limits; a 10-minute-per-problem reasoning CoT is unrealistic for a human contestant. Be explicit about time discipline.
- **Contamination on pre-cutoff contests is extreme.** Problems posted before the model's training cutoff are essentially guaranteed to be in pretraining data (Codeforces problems are discussed on blogs, forums, GitHub, YouTube). Only post-cutoff contests give clean signal.
- **Div 3/4 vs Div 1/2 gap is huge.** A model competing in Div. 3 (easier, weaker field) gets higher percentile numbers than the same model would in Div. 1. Specify.
- **Codeforces problems are non-stationary.** Problem difficulty, contest format, and the contestant population drift over time. Aggregating Elo across 2023 and 2025 contests is noisier than aggregating within a narrow window.
- **Verification is non-trivial.** Codeforces checks submissions via its own judge, which enforces time/memory limits and uses hidden tests. Reimplementing this outside the platform is error-prone; CodeElo's choice to submit directly is the correct one.
- **o1's methodology vs later evaluations differ.** Don't assume o1's 1673 and a 2025 evaluation of R1 are directly comparable without matching methodology. Newer reasoning models report higher Elos partly because benchmarks have loosened (more submissions, more time).
- **Percentile > Elo for public communication.** Elo has no units; percentile is intuitive. Paper tables usually include both.
- **Codeforces is not a pure problem-set benchmark.** It's a live system with partial-credit scoring in some contests, problem categories (algorithms, data structures, math, greedy, DP), and rating updates that depend on who else competed. Reducing this to a single Elo loses detail.

---

## Typical modern numbers

| Model | Codeforces Elo / Percentile |
|---|---|
| o3 (top reasoning, 2025) | >1800 / ~top 1% |
| DeepSeek-R1 | 2029 / ~96th percentile |
| o1 (full) | 1673 / 89th percentile |
| o1-mini (OpenAI blog) | 1650 |
| o1-mini (CodeElo) | 1578 |
| Kimi k1.5 (long-CoT) | 94th percentile |
| QwQ-32B-Preview | 1261 |
| Most non-reasoning LLMs (GPT-4o-class) | <800 / bottom ~25% |

---

## Sources

- Platform: https://codeforces.com.
- Blog: *Learning to Reason with LLMs* — OpenAI, September 2024 — o1 Codeforces results.
- Paper: *CodeElo: Benchmarking Competition-level Code Generation of LLMs with Human-comparable Elo Ratings* — QwenLM, 2025, arXiv 2501.01257 — standardized methodology.
- Paper: *Competitive Programming with Large Reasoning Models* — OpenAI, 2025, arXiv 2502.06807 — o3-era detailed methodology.
- Repo: https://github.com/QwenLM/CodeElo — CodeElo evaluation code.
- Paper: *DeepSeek-R1* — 2025, arXiv 2501.12948 — reports 2029 Elo / ~96.3 percentile.
- Paper: *Kimi k1.5: Scaling Reinforcement Learning with LLMs* — Moonshot AI, 2025, arXiv 2501.12599 — reports 94th percentile (methodology involves majority voting with model-generated test cases, Appendix C.2).
