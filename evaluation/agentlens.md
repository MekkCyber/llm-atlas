# AgentLens
*Depth — one specific benchmark, grounded in its source paper.*

**TL;DR:** Standard coding-agent benchmarks (SWE-Bench, LiveCodeBench) reduce each run to a single pass/fail bit. AgentLens instead scores **the whole trajectory** — instruction following, tool use, self-verification, error recovery, user-facing communication — by pairing formal verification (where an objective check exists) with **LLM-written trajectory reviews** and pairwise comparisons. Runs are used in a nightly pipeline to catch product regressions, not just to rank models.

**Prereqs:** [livecodebench.md](./livecodebench.md), [humaneval.md](./humaneval.md)
**Related:** [../post-training/cot-reward-model.md](../post-training/cot-reward-model.md) · [../agents/README.md](../agents/README.md)

---

## What it is

A benchmark and evaluation harness for **interactive coding agents** — the kind of agents that run multi-turn tool loops (file edit, test, iterate) rather than emit a single code block. The user experience of these agents is a *trajectory*, not an answer, so the benchmark evaluates the trajectory.

AgentLens was released with production use in mind: the same benchmark serves as a nightly-CI regression test for a real agent product, not only as a static leaderboard. Open-source at github.com/agent-lens/agent-lens-bench.

## How it works

Two complementary scoring channels per task:

1. **Formal verifier.** Where an objective check exists — tests pass, file diff matches, output artifact meets spec — a rule-based verifier assigns the ground-truth score. This is the SWE-Bench-style bit.
2. **LLM-judge trajectory review.** A judge LLM produces a **structured, readable review** of the whole trajectory: how well the agent followed instructions, whether it chose the right tools, whether it verified its own work, whether it recovered from mistakes, whether its user-facing messages were clear. The review is a natural-language explanation *paired* with a rubric score, so every run has both a number and a "why."
3. **Pairwise trajectory comparison.** Side-by-side comparison of two agent versions on the same task, judged by an LLM using the same rubric. This is the mode used for regression detection: "is v42's trajectory on task X worse than v41's?"

Reviews and comparisons are stored per run, so failures are triage-able rather than just countable.

## Why it matters

- **Trajectory quality is the deployment quality.** Users of coding agents care about *how* the agent worked, not only *whether* it passed the test. AgentLens is the first widely-scoped benchmark to encode that.
- **Regressions between agent versions are detectable.** Pass rate on SWE-Bench is a coarse aggregate that hides "this version got worse at recovering from failed tests." Trajectory comparison catches it.
- **Reviews are diagnostics.** Because each score comes with a written justification, benchmark output feeds back into training data curation and prompt engineering — the eval doubles as an error-analysis tool.
- **Sets a template.** The trajectory-review-as-eval pattern likely spreads to browser agents, computer-use agents, and research agents — anywhere the "did it work" bit is too coarse.

## Gotchas & tricks

- **Judge-LLM bias.** Reviews reflect the judge's preferences. Rotating or ensembling judges is standard practice; AgentLens is compatible but the paper does not commit to one policy.
- **Rubric drift.** Sub-scores are only comparable across model versions if the rubric is fixed. Any rubric change resets the leaderboard.
- **Pairwise vs pointwise.** Pairwise comparison is more sensitive to small quality shifts than pointwise scoring but requires 2× runs and O(N²) comparisons at scale — the paper triages by comparing only to a reference baseline.
- **Contamination risk.** Trajectory rubrics are novel text; judge LLMs may not have been trained to score them. Warm the judge with a small labeled set before nightly runs.
- **Not a replacement for tests.** The formal-verifier channel is still authoritative for correctness; the LLM-judge channel is complementary, not a substitute for ground truth.

## Sources

- Paper: *AgentLens: Production-Assessed Trajectory Reviews for Coding Agent Evaluation* — Podivilov, Lomshakov, Savin, Startsev, Pozharskiy, Parshin, Nikolenko, 2026 — arXiv:2607.06624.
- Repo: https://github.com/agent-lens/agent-lens-bench
- Related: *SWE-Bench* — Jimenez et al., 2023 — the pass/fail baseline AgentLens critiques.
