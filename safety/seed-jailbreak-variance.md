# Seed-Dependent Jailbreak Variance
*Depth — identically-trained LLMs differ in which jailbreaks break them; jailbreak vulnerability is a property of the training trajectory, not the recipe.*

**TL;DR:** Chang et al. (2026) train 150 copies of a 1B LLM in a controlled factorial — 25 data recipes × 3 pretraining seeds × 2 post-training seeds — and run a jailbreak battery against each. Per-prompt non-refusal rates correlate at **0.984** between identical reruns, drop to **0.811** across pretraining seeds, and drop to **0.779** across data recipes. The "best" attack template stays best after a new pretraining seed in only **27%** of model pairs (vs **67%** after a new post-training seed). A model's per-prompt jailbreak pattern is a fingerprint of its *pretraining run*, not its recipe.

**Prereqs:** [_jailbreaks](_jailbreaks.md), [_attacks](_attacks.md)
**Related:** [safety-case](./safety-case.md), [mismatched-generalization](./mismatched-generalization.md), [sleeper-agents](./sleeper-agents.md)

---

## What it is

The paper directly tests a safety-engineering assumption: that two models trained with the same recipe will fail on the same jailbreaks. If true, a red-team eval on *one* model justifies the recipe. If false, you need seed-ensemble eval to say anything about recipe-level safety.

The result is "false" with strong effect size.

## How it works

### Factorial

- 25 **data recipes** (varying mixture, filtering, decontamination).
- 3 **pretraining seeds** per recipe (same recipe, different init / data order).
- 2 **post-training seeds** per pretraining checkpoint (same SFT + safety recipe, different init / order).
- Total 25 × 3 × 2 = 150 models. All 1B parameters, same architecture.

### Jailbreak battery

A set of standard attack templates (prefix injection, roleplay, encoding tricks, low-resource language, etc. — see [_jailbreaks](_jailbreaks.md)) applied to a shared prompt set. Measure per-prompt non-refusal rate per model.

### Correlation panel

For each pair of models, compute Pearson correlation of per-prompt non-refusal vectors. Group pairs by what differs:

| Pair type | Per-prompt correlation |
| --- | --- |
| Identical pretraining + identical post-training (rerun) | 0.984 |
| Different post-training seed | ~0.90 |
| Different pretraining seed | **0.811** |
| Different data recipe | **0.779** |

### Attack-template stability

For each model, rank attack templates by success rate. Then across model pairs, compute P(best template stays best after a change):

| Change | P(best attack stays best) |
| --- | --- |
| Different post-training seed | 67% |
| Different pretraining seed | 27% |

### Fingerprinting

Given a model's per-prompt pattern, a nearest-neighbour classifier recovers *which pretraining run produced it* with high accuracy.

## Why it matters

- **Breaks single-model red-teaming as a safety case.** A "frontier model passes this battery" claim doesn't generalize to the *next* training run from the same recipe. Red-team eval should be seed-ensemble.
- **Separates recipe-level safety from instance-level safety.** Two different problems: "is this recipe safe in expectation?" (needs many seeds) vs "is this deployed model safe?" (needs instance-level eval on *this* checkpoint).
- **Argues against overfitting to one attack set.** Because the top attack changes with pretraining seed, red-teams that tune against a fixed battery underestimate how much the next seed will shift.
- **Implication for safety-cases.** A [safety-case](./safety-case.md) built on jailbreak eval of one checkpoint needs an explicit "seed sensitivity" argument.

## Gotchas & tricks

- **1B scale is small.** The variance may shrink at frontier scale — or it may grow, since harder prompts are more sensitive. Scale extrapolation is open.
- **"Post-training seed" here is weaker than "post-training recipe."** The paper fixes the safety recipe and varies only the stochastic seed; a different safety recipe would move more.
- **The battery is fixed.** If the battery is itself overfit to a generation of attacks, the 27% stability figure is partially an artifact of battery choice. Still, the delta between seed groups is the signal.
- **Fingerprinting is dual-use.** It is a nice diagnostic for reproducibility but also a way to attribute a leaked model to a specific training run.
- **Not a mitigation.** The paper diagnoses the problem; it doesn't fix it. "Train with safer recipes" is still the right move — just don't accept one model's pass as evidence.

## Sources

- Paper: *Identically Trained Language Models Break on Different Jailbreaks* — Chang, Kuo, Lee, Tsai — Cerulean Labs, 2026 — [arXiv:2610.34074](https://arxiv.org/abs/2610.34074).
- Related framing: [safety-case](./safety-case.md), [_jailbreaks](_jailbreaks.md), [_attacks](_attacks.md).
