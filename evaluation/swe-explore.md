# SWE-Explore
*Depth — a benchmark for the repository-exploration sub-skill of coding agents.*

**TL;DR:** A benchmark that **isolates repository exploration** out of end-to-end SWE-bench-style evaluation. Given a repo + an issue, the agent returns a **ranked list of relevant code regions under a fixed line budget**. **848 issues, 10 programming languages, 203 open-source repositories**, with **line-level ground truth distilled from the trajectories of independent agents that successfully solved the same issue**. Evaluated along *coverage*, *ranking*, and *context-efficiency* axes. Introduced by Zhang et al. (SJTU / Xinjiang U. / UIUC / CUHK), 2026 (arXiv 2606.07297).

**Prereqs:** [humaneval.md](humaneval.md)
**Related:** [livecodebench.md](livecodebench.md) · [codeforces-benchmark.md](codeforces-benchmark.md)

---

## What it is

SWE-bench grades coding agents on the *outcome* (did the patch land?). SWE-Explore grades the **first stage that produced that outcome**: localization. Most SWE-bench failures are localization failures — the agent never read the right code — but localization quality is invisible in the binary resolved/unresolved metric.

The benchmark asks one narrow question: given an issue, which code regions are relevant? Output is a ranked list; budget is fixed in lines (so the metric is about precision and ordering under a real constraint).

Stats:
- **848 issues** drawn from real OSS bug tracker / PR data.
- **10 languages**.
- **203 repositories**.
- **Line-level ground truth** — not file-level, not function-level.

## How it works as an LLM eval

- **Behavioral ground truth.** For each issue, run several independent solver agents that successfully resolved it. Log the code regions each solver actually consulted. The intersection / union of those regions is the line-level relevance label — what the *successful behavior* used, not what a human annotator guessed.
- **Three metrics:**
  - *Coverage*: of the labeled regions, how many appear in the returned ranked list?
  - *Ranking*: are they near the top of the list?
  - *Context-efficiency*: how many irrelevant lines per relevant line?
- **Tiers compared.** Classical retrieval (BM25, embedding KNN), general coding agents (zero-shot prompting), and specialized localizers (purpose-built tools).

## Why it matters

- **Disaggregates SWE-bench.** If you can grade localization separately, you can optimize it separately — and plug the best localizer into any patch generator.
- **File-level is largely solved.** Modern retrievers find the right files. The paper's finding: **line-level coverage and efficient ranking remain the differentiators** for SOTA.
- **Strongly tracks downstream.** Exploration metrics correlate with downstream patch quality, so SWE-Explore is a cheap predictor of SWE-bench-style outcomes.
- **Reuses existing trajectories.** No new human labeling round — the labels come from existing solver agents. Cheap to extend as solvers improve.

## Gotchas & tricks

- **Ground truth is behavioral.** Code regions that *would have helped* but no successful solver consulted are not labeled — the metric is biased toward the paths past solvers took.
- **Budget choice matters.** Tight line budget rewards precision; loose budget rewards recall. Compare like-for-like budgets across papers.
- **Language imbalance.** 10 languages, but the heavy-tail distribution of repos means some languages have far more issues. Per-language reporting matters.
- **Trajectory selection bias.** The "independent successful solvers" cohort tilts toward the kinds of issues current agents can solve. The benchmark inherits that bias.
- **Not a replacement for SWE-bench.** Exploration is necessary, not sufficient — perfect localization with bad patch generation still fails. Use SWE-Explore alongside SWE-bench, not instead of.

## Sources

- Paper: *SWE-Explore: Benchmarking How Coding Agents Explore Repositories* — Zhang, Wang, Liang, Shi, Zeng et al. — SJTU / Xinjiang U. / UIUC / CUHK, 2026 — arXiv 2606.07297.
- Reference: *SWE-bench* — Jimenez et al., 2024 — the end-to-end coding-agent benchmark SWE-Explore disaggregates.
