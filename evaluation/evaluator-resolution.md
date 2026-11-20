# Evaluator Resolution (Self-Improvement Bottleneck)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Self-improving agent loops (propose candidate → evaluate → select winner → iterate) stall not because methods fail to propose better candidates, but because **evaluators are too noisy to reliably rank true improvements**. The Resolution Hypothesis frames the loop as a signal-vs-noise detection problem: resolution = the evaluator's ability to distinguish better from worse candidates. Empirical support at both candidate-level and method-level: when resolution is low, even clearly better candidates fail to be selected, and progress stalls regardless of proposer quality.

**Prereqs:** [README.md](README.md)
**Related:** [../post-training/reasoning/prm.md](../post-training/reasoning/prm.md), [../post-training/reasoning/orm.md](../post-training/reasoning/orm.md)

---

## What it is

Self-improving agents (AutoGPT-style, OpenHands, Sakana AI Scientist, agent-evolution papers) run a loop: propose a change → measure outcome → keep if better. The standard failure mode is "stalling": iteration count grows without quality improving.

The Resolution Hypothesis attributes this stalling to the *evaluator* — the measurement step — being too noisy to produce a reliable signal, rather than the proposer failing to generate better candidates.

---

## How it works

Model each iteration as a hypothesis test: candidate $A$ (current) vs candidate $B$ (proposed).

- **True effect**: whether $B$ is actually better than $A$ on the target objective.
- **Observed effect**: the evaluator's score difference, which is noisy.

The evaluator's **resolution** is its ability to correctly rank $B$ above $A$ when $B$ is truly better. Formally, a classifier AUROC between "true improvement" and "regression" given the evaluator's reported signal.

**Empirical claim.** Across agent-evolution systems, the authors find:

- Low-resolution evaluators select randomly — proposer quality becomes irrelevant.
- At typical published eval-noise levels, even method-level improvements (changes to proposer design) are indistinguishable at single-run scale.
- Variance reduction on the evaluator (more samples, calibrated scoring) produces larger gains than better proposers.

---

## Why it matters

- **Reorients self-improvement research.** Fix the evaluator before fixing the proposer. This inverts a long-running priority in agent-evolution work.
- **Explains a reproducibility headache.** Agent-evolution results are famously hard to replicate — the Resolution Hypothesis predicts that most reported "improvements" at single-run comparison are within evaluator noise, hence not real signal.
- **Transfers to inference-time search.** Any search-with-verifier setup (CoT + PRM, best-of-N with a reward model) has the same bottleneck: verifier resolution bounds search quality. See [prm.md](../post-training/reasoning/prm.md).

---

## Gotchas & tricks

- **Resolution has a cost.** Evaluation runs money and time; higher resolution usually means more samples, more judges, or more diverse benchmarks. Budget accordingly.
- **Calibration matters.** Noisy but *calibrated* evaluators can be averaged into reliable signals; miscalibrated evaluators can't.
- **LLM-as-judge inherits LLM biases.** Judge models have their own preferences (length, style). Resolution on *your* axis of improvement may be low even when the judge seems confident.
- **Method-level comparisons need many seeds.** The paper argues most agent-evolution ablations don't run enough seeds to beat evaluator noise. Treat single-seed deltas with suspicion.

---

## Sources

- Paper: *The Resolution Hypothesis: Agent Evolution's Key Is Identifying Improvements* — Zeng, Lan, Lu, Yan, Lin, Xu, Luo, Wang, Li, Shan, Ren (ShanghaiTech) — 2026 — [arXiv:2611.23068](https://arxiv.org/abs/2611.23068).
- Related: *Process Reward Models* literature (Lightman 2023, Uesato 2022) — same resolution problem in CoT verification.
