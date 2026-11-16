# SO-Arena
*Depth — a benchmark for System One decision LLMs across capability and reliability.*

**TL;DR:** "System One" models — LLMs tuned to answer in zero or few generated tokens for low-latency decisions — are showing up inside agent loops (routing, retrieval, planning), but no public benchmark isolated their *reliability* separately from raw accuracy. SO-Arena covers 11 domains and 5 capability families, with two orthogonal axes: capability (evidence-grounded tiers + interactive sandbox) and reliability (decision stability under option reordering + recognizing "no correct answer"). The best capability-ranked model among 21 evaluated placed only 7th on reliability.

**Prereqs:** [mmlu.md](mmlu.md)
**Related:** [ifeval.md](ifeval.md)

---

## What it is

A benchmark for LLMs used as fast, low-token decision engines. Unlike MMLU-style evals that measure accuracy on multiple-choice, SO-Arena is explicitly two-axis: a model can be capable and unreliable, or reliable and incapable, and the benchmark exposes both.

## How it works

**Capability axis.** 11 domains × 5 capability families. Two task types:
- *Evidence-grounded multiple-choice*, split into difficulty tiers so the aggregate score isn't dominated by easy questions.
- *Interactive sandbox tasks* where the model must make a decision in a small simulated environment. Each capability family has its own sandbox.

**Reliability axis.** Two probes, applied on top of the capability questions:
- *Option-reordering stability.* The same question is re-asked with options permuted. A reliable model answers consistently; unreliable ones flip.
- *"None of the above" recognition.* A fraction of questions have no correct option. The model must identify that fact rather than picking the least wrong.

Scores on the two axes are reported separately and ranked separately. The headline finding is that the two rankings diverge sharply.

## Why it matters

As low-latency decision models get embedded inside agent stacks, stability under adversarial option framing is a first-class deployment property — a model that flips its answer when options are reordered is a liability inside a pipeline. SO-Arena gives a shared, quantitative handle on this, and pins down a measurable capability-vs-reliability tradeoff that practitioners have described anecdotally for a while.

## Gotchas & tricks

- Reliability metrics are *paired*: option-reordering is a within-question property and needs multiple forward passes per item. Budget cost accordingly.
- "No correct answer" items have to be mixed in at a low rate — a model that recognizes a 50% rate learns the meta-rule rather than the skill.
- Capability rank and reliability rank should be read side-by-side; collapsing to one score (even a weighted combination) defeats the point of the benchmark.
- Difficulty tiers inside a domain are not comparable across domains — a "tier 3" in one domain can be easier than "tier 2" in another. Aggregate within-tier, within-domain first, then compare.
- Sandbox tasks are seeded. Running them with the same seed across models is essential for fair reliability comparison; re-seeding each run mixes noise into the metric.

## Sources

- Paper: *SO-Arena: Benchmarking the Capability Spectrum and Reliability of System One Models* — 2026 — [arXiv:2611.15777](https://arxiv.org/abs/2611.15777)
