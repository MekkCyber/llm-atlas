# SO-Arena — Benchmarking System One Decision Models
*Depth — benchmark for single-forward-pass "System One" decision models.*

**TL;DR:** A 2,919-task benchmark for *System One* decision models — the single-forward-pass, structured-output mode of an LLM that returns a typed choice (and its probabilities) without emitting any reasoning tokens. Covers 11 domains and five capability families, adds difficulty tiers and interactive sandbox evaluation, and scores a dedicated **reliability** axis: decision stability under option reordering, and recognition of "no correct option is offered". Across 21 models, the capability leader ranks only seventh in reliability — SO-Arena surfaces a capability-reliability gap that generic LLM benchmarks miss.

**Prereqs:** [mmlu](mmlu.md), [ifeval](ifeval.md)
**Related:** [_post-training](../post-training/_post-training.md)

---

## What it is

A benchmark for **System One decision models** — models used in a mode where one forward pass maps an input to a typed output (a labeled choice with calibrated probabilities), with no intermediate chain-of-thought. This is the mode agent harnesses and routers call when they want a cheap, structured decision.

The benchmark has 2,919 tasks across 11 domains and five capability families. Reference labels are checked against model-visible inputs. Tasks have difficulty tiers, some run in an interactive sandbox, and each task has a System Two (reasoning) comparison point.

## How it works

Two axes are scored independently:

- **Capability** — accuracy at picking the labeled correct option.
- **Reliability** — two sub-scores:
  - *Decision stability*: scramble the option order and measure whether the model's choice changes.
  - *No-answer recognition*: on tasks where no option is correct, does the model abstain?

System One is compared against System Two (full CoT) on the same tasks — a controlled measure of what reasoning actually buys you per domain.

## Why it matters

Agent systems increasingly call LLMs in "structured output" mode for routing, tool selection, and policy choices. The dominant eval modes (MMLU-style accuracy, pairwise preference) don't capture the two things that actually break in production: decisions that flip when the order of options changes, and failure to abstain when nothing fits.

SO-Arena gives these first-class metrics and shows the ranking under reliability is *not* the ranking under capability — so picking a model for a decision slot needs its own eval, not borrowed LLM leaderboard numbers.

## Gotchas & tricks

- "System One vs System Two" here is a *mode* of the same model, not two models. The same checkpoint can run in either mode and the benchmark compares both.
- Stability under option reordering is distinct from chance-corrected accuracy. A model that answers deterministically but wrongly is 100% stable; the metric is about *robustness to irrelevant surface changes*, not correctness.
- The 21-model result that the capability leader is 7th in reliability makes "does it listen to its own options" a model-selection criterion, not a free bonus.

## Sources

- Paper: *SO-Arena: Benchmarking the Capability Spectrum and Reliability of System One Models* — arXiv 2611.15777 — https://arxiv.org/abs/2611.15777
