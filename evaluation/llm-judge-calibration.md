# LLM-Judge Calibration (LUCID)
*Depth — prompt-based self-correction to debias an LLM step-rater using a tiny expert-labeled set.*

**TL;DR:** Cheap LLM raters grade reasoning steps at scale, but carry systematic biases (e.g., GPT-5-mini is too strict and under-rates computation/verification/late-trace steps). LUCID (2026) diagnoses the bias from a small expert-labeled sample, writes the bias into a plain-text prompt, and asks the rater to re-grade — no retraining, no weight access. Pair with Design-based Supervised Learning (DSL) for calibrated regression and transition estimates. Cuts GPT-5-mini's calibration gap on PRM800K from 7.8 points to 0.9 in one pass.

**Prereqs:** [README](README.md), [../post-training/reasoning/prm](../post-training/reasoning/prm.md)
**Related:** [../post-training/cot-reward-model](../post-training/cot-reward-model.md)

---

## What it is

A calibration loop for LLM-as-judge step raters. It converts an expert-labeled sample of true step-level "sound/unsound" labels into (a) a natural-language description of the rater's bias and (b) a reprompting instruction that the rater applies when re-grading the full corpus.

## How it works

LUCID = diagnosis + prompt self-correction + DSL.

1. **Expert sample.** Collect a small set of step ratings labeled by domain experts (as few as 1,000 steps suffices).
2. **Diagnose.** Compare the rater's ratings on this set to the expert labels. Summarize systematic patterns in plain text (per-step-type, late-vs-early, strictness direction).
3. **Reprompt.** Attach the bias summary to the rater's prompt and have it re-grade unseen steps. This is the core loop.
4. **DSL pooling.** Combine raw rater scores with expert labels using Design-based Supervised Learning — doubly-robust estimators that give calibrated regression coefficients and sound-step transition probabilities.

Optional pre-/post-check: skip reprompting if the rater is already calibrated (saves cost), and roll back reprompts that didn't help (safety).

## Why it matters

Three practical wins for reasoning evaluation:

- **No retraining, no weight access.** Works with frontier closed-weight raters.
- **Bias profiles transfer.** A GPT-5-mini-derived bias profile lifts GLM-5.2 accuracy, suggesting shared failure modes across raters.
- **DSL gives principled aggregation.** For small label budgets, doubly-robust DSL beats expert-only regression and cuts sound-step transition error by ~⅔.

Likely composes with PRM-based RL: the same bias-correction signals can feed back as reward-shaping corrections rather than just evaluation.

## Gotchas & tricks

- **The bias summary must be concrete.** Vague feedback ("be less strict") does not improve the rater; the paper's effective prompts cite per-step-type and per-trace-position biases.
- **Prompt corrections shift thresholds, not discrimination.** Class-balanced sound/unsound separation (AUC-like) is not improved by reprompting; what changes is where the threshold is set. For truly bad raters, LUCID won't help.
- **One-pass is usually enough.** Iterated reprompts risk overcorrection; the paper's pre-/post-checks catch this.
- **DSL assumes the expert labels are unbiased** (they're the ground truth by construction). Noisy expert labels propagate into DSL estimates.
- Domain transfer is good but not perfect: calibrating on math may undercorrect for web/logic benchmarks.

## Sources

- Paper: LUCID: Statistically Grounded Self-Correction for LLM Reasoning Evaluation — Murugaboopathy, Daoud & Jerzak, 2026 — [arXiv:2611.31595](https://arxiv.org/abs/2611.31595)
