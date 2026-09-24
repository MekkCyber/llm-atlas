# Taste-Bench
*Depth — fork-mining benchmark for long-horizon agent decision quality.*

**TL;DR:** Taste-Bench is a benchmark that measures an agent's *taste* — how well it makes mid-run decisions at forks in a long-horizon trajectory — decoupled from end-to-end success. Fork questions are mined **automatically** from parallel attempts at the same task and detour segments within single trajectories: pick the branch, evaluated on branches whose outcomes differ. Frontier models max out around 59.7% accuracy; distilled outcome-aware taste transfers to held-out SWE-bench Pro tasks.

**Prereqs:** [README.md](./README.md), [../agents/README.md](../agents/README.md)
**Related:** [../post-training/rejection-sampling.md](../post-training/rejection-sampling.md)

---

## What it is

An evaluation focused on the mid-trajectory decision quality of long-horizon agents (engineering, research). End-to-end pass/fail metrics conflate many decisions into one number; Taste-Bench isolates each fork so a model's decision quality is measurable separately from its execution quality.

Fills a specific evaluation gap: existing agent benchmarks (SWE-bench, GAIA, ProgramBench) score final task success but say nothing about *which* mid-run decisions distinguished the strong runs from the weak ones.

## How it works

**Mining forks:**

- Take many trajectories (parallel attempts on the same task, plus detour segments where a single trajectory backtracked). At each fork point, multiple branches were tried; some led to a better outcome than others.
- Keep forks where the pre-fork context is identical and the outcomes measurably differ. Discard ties.
- Automatic — no human annotation required.

**Presenting a fork to the evaluated model:**

- Show only the pre-fork context (everything up to the decision point).
- Show the candidate directions available at the fork (typically 2–4).
- The model picks one *without* seeing what happens downstream. Correct = the branch that led to the better outcome in the source trajectory.

**Difficulty stratification:**

- Forks whose deciding evidence appears **later** in the trajectory are much harder (the pre-fork context under-specifies which branch wins). Every tested model suffers on late-evidence forks.
- Extra reasoning budget doesn't help — the deciding information isn't in the pre-fork context to begin with.

**Training taste (bonus contribution):**

The paper also shows taste is *trainable*: distill an outcome-aware teacher (a model that saw the downstream branch outcomes) into a student that only sees the pre-fork context. The student improves both Taste-Bench and downstream SWE-bench Pro end-to-end success.

## Why it matters

- **Decouples decision quality from execution.** Strong models can look weak on end-to-end benchmarks because of execution bugs; Taste-Bench separates the two.
- **Cheap to grow.** Fork mining is fully automatic — any agent that produces logged parallel attempts contributes examples.
- **Training signal, not just a metric.** Outcome-distilled taste transfers off-benchmark, giving a concrete SFT/distillation objective for agent decision quality.

## Gotchas & tricks

- **Source-trajectory bias.** The "correct" branch is whichever won in the source trajectory. If the source agents are systematically wrong on some fork type, that shows up as biased ground truth. Sample from diverse source agents.
- **Late-evidence forks are noisy.** By construction, no model can reliably answer them. Reporting overall accuracy without stratifying by evidence-position hides where the signal is.
- **Reasoning budget doesn't help** — a surprising finding worth flagging when comparing models with different thinking budgets on this benchmark.
- **Distillation shortcut.** Once outcome-aware teachers exist, students can be tuned to inherit taste directly — meaning frontier-model gaps on Taste-Bench may close through distillation rather than base capability.

## Sources

- Paper: *The Tasteful Agent: Measuring and Improving Taste in Long-Horizon Tasks* — Pan et al. (Microsoft / CityU HK), 2026 — [arXiv:2609.25804](https://arxiv.org/abs/2609.25804).
