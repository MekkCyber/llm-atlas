# Early Outcome Prediction (EarlyEval)

*Depth — halting an agent evaluation run the moment a lightweight classifier is confident of success or failure.*

**TL;DR:** Frontier-model agent benchmarks (SWE-bench, TerminalBench, Toolathlon) cost hundreds to thousands of dollars per pass, and iterative development amortizes that many times over. Benchmark distillation cuts task counts but leaves per-task execution untouched. **EarlyEval** (Shi et al., 2026) adds a complementary axis: *within* a task, train a pair of LightGBM success and failure classifiers on behavioral, textual, and reference-solution features and stop the run as soon as either crosses a calibrated confidence threshold. Across three benchmarks: **13–26% of agent steps eliminated**, **up to 44.1% input / 29.4% output tokens saved**, at 89–97% classifier accuracy, with only 1–2 pp change in resolve rate.

**Prereqs:** [../agents/README](../agents/README.md), [README](README.md)
**Related:** [../inference/README](../inference/README.md)

---

## What it is

A wrapper around an agent evaluation loop that predicts the eventual outcome from each step's intermediate state and halts the run when the prediction is confident enough. Orthogonal to benchmark distillation (which trims tasks) — EarlyEval trims *time inside each task*.

## How it works

### Two classifiers, not one

EarlyEval trains **two separate LightGBM classifiers**:

- A **success predictor** that fires once the agent's trajectory is confidently on a path to task completion.
- A **failure predictor** that fires once the trajectory is confidently doomed (loops, giving up, wrong artifact created).

Both are consulted every step. Either passing its confidence threshold halts the run and locks in the predicted outcome.

Why two: success and failure paths look different structurally, and forcing one classifier to model both wastes capacity. Separate models each optimize their own halt condition.

### Features

Cheap-to-compute, no LLM in the classifier:

- **Behavioral** — number of tool calls, sequence of tool types, retries, time per step, action-type histogram.
- **Textual** — surface features of the agent's messages and outputs (length, error-string presence, keyword hits).
- **Reference-solution** — for benchmarks with a known solution, similarity between the agent's current artifact and the target (e.g. AST distance for code, diff overlap for patches).

LightGBM was chosen because features are tabular and per-step cost is negligible next to any agent step it would halt.

### Confidence calibration

Confidence thresholds are calibrated on a held-out set so that early halts introduce ≤ 1–2 pp change in the per-agent resolve rate. Higher thresholds → smaller savings but stronger accuracy match; the paper reports operating points at 89–97% classifier accuracy.

### Results shape

- **SWE-bench Verified**: substantial step reduction with token savings dominated by long-running compile/test loops.
- **TerminalBench**: shorter tasks, so relative token savings are smaller but step-count savings are largest.
- **Toolathlon**: multi-turn tool workflows benefit most from cutting doomed runs early.

Across all three: 13–26% fewer steps, 44.1% input / 29.4% output tokens saved at the top-line accuracy operating point.

## Why it matters

- **Doubles or triples the iteration rate on agent benchmarks.** Dev cycles on SWE-bench take hours; cutting half the tokens roughly doubles the number of ablations a lab can afford.
- **Composes with distillation.** Pruning tasks (Mini-SWE-bench) and pruning steps within tasks (EarlyEval) multiply — a 3× cost reduction from combining them is realistic.
- **Cheap enough to be always-on.** The classifier is CPU-side LightGBM with tabular features — no GPU cost, no LLM call. There is no reason not to run it if the benchmark is expensive.
- **Enables tighter agent-training feedback loops.** Cheaper eval means eval-during-training becomes viable, not just eval-after-training — a substrate for RL-on-agents work.

## Gotchas & tricks

- **Classifier drift across models.** A classifier trained on one agent's traces underperforms on a different agent (different tool-call patterns). Retrain when the agent scaffold changes.
- **Feature leakage risk.** The reference-solution features are per-benchmark; using them in training and reporting savings on the *same* benchmark is fine, but transferring the classifier to a new benchmark without reference-solution features requires care.
- **Failure-side biases.** The failure classifier is more sensitive to loops than to silent wrong answers. If a benchmark's failure modes are quiet (e.g. subtly wrong code that passes the agent's own tests), the failure classifier under-triggers.
- **Halt-early = lose diagnostic signal.** An early-halted run gives no post-hoc trace of what happened after the halt point. If the eval is also being used for error-analysis, run without EarlyEval on a small held-out slice.
- **Do NOT deploy the classifier as a runtime scaffold.** It's calibrated for *evaluation efficiency*, not for user-facing decisions — an early-halt in production is a dropped user request.

## Sources

- Paper: *EarlyEval: Cheaper Agent Evaluation via Early Outcome Prediction* — Yuling Shi, Zhensu Sun, Junsen Dong, Chengcheng Wan, David Lo, Xiaodong Gu — 2026 — [arXiv:2609.02783](https://arxiv.org/abs/2609.02783) — SJTU · Singapore Management U. · ECNU.
