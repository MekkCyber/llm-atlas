# Calibrated Detector (Jev)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A small model RL-post-trained to output **calibrated probabilities** on typed yes/no questions about a piece of text, used as a **zero-shot detector** for a wide taxonomy of AI misuse (sycophancy, jailbreaks, deception, prompt injection, hallucination, privacy leakage, social bias, reward hacking, hidden uncertainty, power-seeking). One generic question suffices — the paper's model *Jev* reaches median **AUROC 0.886** zero-shot across 44 benchmarks and five target models, beating supervised baselines on most and matching human-referee agreement at much lower cost than LLM-judge alternatives.

**Prereqs:** [_rl](../post-training/_rl.md), [cot-monitoring](cot-monitoring.md)
**Related:** [_attacks](_attacks.md), [_jailbreaks](_jailbreaks.md), [_rewards](../post-training/_rewards.md)

---

## What it is

Standard misuse detectors are either brittle supervised classifiers (one per failure type) or expensive LLM judges (frontier API call per input). Jev is a third path: a single small model whose *outputs are calibrated*, so a single generic yes/no probe transfers across misuse categories.

## How it works

- **Task format.** Given text $x$ and a *typed question* $q$ (e.g. "Is this response sycophantic?"), Jev returns a probability $p(q \mid x) \in [0,1]$. Multiple typed questions can be answered in a single forward pass.
- **RL objective.** The reward penalizes miscalibration — for a rollout batch, compute a proper scoring rule (e.g. Brier or log score) against observed frequencies of the yes/no ground truth. This is not accuracy-only RL; the reward is optimized when the model's probabilities *match true frequencies*, not when they are extreme.
- **Generic zero-shot probe.** At inference, ask a single generic question ("Is this output problematic in any way?") — because calibration generalizes across question types more cleanly than accuracy, this transfers to unseen failure categories without retraining.
- **Benchmark.** *RLCDAlignBench* — 10 alignment-failure types × multiple attack styles per type; 44 sub-benchmarks; five target models under evaluation.

## Why it matters

- **One small model replaces a zoo of specialized detectors.** Practical building block for agent-side safety monitoring and cheaper than always-on frontier-API guardrails.
- **AUROC 0.886 median** with a *generic* zero-shot question — comparable to supervised baselines that were built per failure type.
- **Calibration matters more than accuracy at deployment.** A calibrated 0.7 lets you trade off recall vs. false-positive rate at operating time; a hard classifier doesn't.

## Gotchas & tricks

- The reward function is a *proper scoring rule* — don't substitute plain accuracy or the model will collapse to overconfident outputs.
- Calibration depends on the reference distribution used during RL — deploy-time calibration on out-of-distribution inputs degrades gracefully but drifts.
- Generic-question zero-shot is a *median* result — for specific attack styles (e.g. very subtle prompt injection), a typed question still beats the generic probe.
- Runs at much lower cost than an LLM-judge — but is not free; a small quantized version can be the difference between "add to every agent step" and "batch-sample later."
- Batch multiple typed questions in one call to amortize the forward pass — Jev outputs per-question probabilities in a single generation.
- Recalibrate periodically on production traffic; RL-time calibration is on a fixed mixture and drift is real.

## Sources

- Paper: *Just Ask Jev: Reinforcement Learning for Calibrated Decisions as a Zero-Shot Detector of AI Alignment Failures* — Guo et al., NTU / UNSW, 2026 — [arXiv:2609.29429](https://arxiv.org/abs/2609.29429).
