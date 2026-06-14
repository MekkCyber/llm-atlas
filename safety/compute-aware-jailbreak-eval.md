# Compute-Aware Jailbreak Evaluation
*Depth — measuring adversarial robustness as a curve of attack success rate vs. cumulative attacker FLOPs, rather than a single fixed-budget ASR.*

**TL;DR:** Standard adversarial-robustness reports give one number: **attack success rate (ASR) at a fixed query budget**. This hides the fact that different attacks cost orders of magnitude different compute — a 25% ASR via expensive optimization is not the same threat as a 25% ASR via cheap prompt engineering. Replace the single number with **ASR(FLOPs)**: cumulative attacker compute on the x-axis, ASR on the y-axis. Different attack families and different models become directly comparable.

**Prereqs:** *(none)*
**Related:** [_jailbreaks.md](_jailbreaks.md) · [_attacks.md](_attacks.md) · [../evaluation/README.md](../evaluation/README.md)

---

## What it is

An evaluation methodology for LLM adversarial robustness where the metric is a *curve* rather than a point. The x-axis is cumulative attacker FLOPs (or wall-clock GPU-seconds as a proxy); the y-axis is ASR. Each attack family traces a curve; each model is characterized by the *family* of curves under attack.

Replaces the conventional reporting:
- Old: "Model M has ASR=25% on attack A at budget B."
- New: "Model M's ASR(FLOPs) curve for attack family A reaches 25% at 10^15 FLOPs, 60% at 10^17 FLOPs."

---

## How it works

### Measuring attacker FLOPs

For each attack, the methodology accumulates the FLOPs spent across all queries, search steps, model inferences (including auxiliary models used by the attacker), and post-processing. The accounting is principled — every operation that costs the attacker compute is included.

### Per-attack curves

An attack family (prompt-engineering, GCG-style optimization, transfer attacks) is run with increasing budgets. The resulting per-attack ASR(FLOPs) curve characterizes the attack's *efficiency* in addition to its peak success.

### Cross-model comparison

Two models with the same fixed-budget ASR but different ASR(FLOPs) curves are differently robust: the model whose curve is shifted right is harder to attack per FLOP.

### Cross-attack comparison

Within a model, different attack families produce different curves. The attacker's actual best strategy is the *upper envelope* across families at each FLOP level.

---

## Why it matters

- **Honest threat modeling.** Defenders care about the *cost* of breaking the model, not just whether it's breakable in principle.
- **Comparable across attack families.** Cheap and expensive attacks now share an axis. Stops fixed-budget ASR from making expensive optimization look "as bad as" cheap prompt engineering.
- **Composes with capability evals.** Capability benchmarks already report compute on the x-axis (training/inference FLOPs); robustness eval now matches that idiom.
- **Reveals robustness mismatches.** Models that look identical at fixed budget can be very different at high pressure — a fact the old metric hides.

---

## Gotchas & tricks

- **FLOP accounting is imperfect.** Different hardware utilizes FLOPs differently; report both raw FLOPs and a wall-clock estimate.
- **Some attacks have non-monotonic curves.** A bad search can plateau or even degrade with more compute; the curve is the *best* result up to each budget, not a single run.
- **Reporting the curve is expensive.** Defenders running broad robustness sweeps must accept the cost; for headline numbers, pick a few canonical FLOP levels (e.g. 10^14, 10^16, 10^18).
- **Doesn't cover model-stealing attacks.** Attacks that exfiltrate the model can amortize FLOPs across many later attacks; the framework is per-instance.

---

## Sources

- Paper: *Compute-Aware Evaluation of Adversarial Robustness in Language Models* — Ehghaghi et al., U. Toronto + Vector + HF, 2026 — [arXiv:2606.11409](https://arxiv.org/abs/2606.11409).
