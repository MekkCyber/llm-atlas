# Linear Probes

*Depth — training a linear classifier on top of frozen LLM activations to test whether a target property is linearly decodable.*

**TL;DR:** Take frozen activations from some layer of an LLM. Train a logistic-regression-class linear classifier on labeled examples of a property (e.g. "is the model deceiving"). High probe accuracy = the property has a linear representation in that layer. Probes are the cheapest and most widely-used interpretability technique — but they overfit easily, and "probe accuracy" is not the same as "the model uses this direction".

**Prereqs:** [../fundamentals/attention.md](../fundamentals/attention.md)
**Related:** [deception-probes.md](./deception-probes.md), [../safety/cot-monitoring.md](../safety/cot-monitoring.md), [../safety/scheming.md](../safety/scheming.md)

---

## What it is

A probe is a small classifier (almost always linear: $y = \sigma(w^\top h + b)$) trained on activations $h$ from a frozen base model. The setup:

1. Pick a layer $\ell$ and a token position $t$ in the LLM. Extract $h_{\ell, t}$.
2. Collect labeled examples $(h_{\ell, t}^{(i)}, y^{(i)})$ for some binary property $y$ (lying / truthful, refusal / compliance, in-distribution / OOD).
3. Train a linear classifier on those activations. Report AUROC on held-out data.

Probes do **two** things — and people regularly conflate them:

- **Decodability claim:** "the model's activations contain enough information to predict $y$." Strong probes prove this.
- **Causal / use claim:** "the model *uses* this direction to drive $y$-related behavior." Probes alone never prove this — you need activation patching or steering.

## How it works

Standard recipe:

```
for layer ℓ in candidate_layers:
    H_train = activations(model, prompts_train, layer=ℓ, position=t)
    H_test  = activations(model, prompts_test,  layer=ℓ, position=t)
    w, b = logistic_regression(H_train, y_train)
    auroc = AUROC(σ(w · H_test + b), y_test)
```

Variants:
- **Multi-dim probe.** Replace the scalar score with a $k$-dim linear projection + classifier head. Tests whether the property occupies a subspace rather than a single direction.
- **Permutation null.** Refit the probe on label-shuffled data; the gap between true AUROC and null AUROC is the real signal.
- **Cross-domain transfer matrix.** Train on domain A, test on domain B. Off-diagonal collapse = the probe is fitting domain features, not the property.
- **Entropy residualization.** Regress out the activation's L2 norm or attention entropy first; tests whether the probe is just reading a generic confusion signal.

## Why it matters

- **Cheap monitor.** Probes are 1–2 orders of magnitude cheaper to train and run than SAEs or activation patching. The right first tool for any "is property X representable" question.
- **Safety / deception monitoring proposals.** Linear deception probes are being proposed as automated monitors that score whether a generation is deceptive without ground-truth labels — but see [deception-probes.md](./deception-probes.md) for the robustness story.
- **Subspace evidence beats direction evidence.** Modern work (Gemma 3 probe pressure-tests, 2026) rejects the single-direction hypothesis for many properties — properties live in multi-dim subspaces, not on a line.

## Gotchas & tricks

- **High clean AUROC ≠ deployable.** AUROC > 0.99 on a clean benchmark routinely drops below 0.7 under stylistic shift. Always evaluate cross-domain.
- **Style augmentation rescues robustness.** Training probes on prompts varied across writing styles (formal / casual / poetic / technical) recovers most of the robustness gap on unseen styles.
- **Position matters.** Last-token probes ≠ assistant-turn-end probes ≠ early-CoT probes. Each measures something different.
- **Probes don't prove causation.** A probe that scores deception with AUROC 0.998 doesn't mean ablating that direction reduces deception — and often it doesn't.

## Sources

- *Pressure-Testing Deception Probes in LLMs* — Kumar, 2026 — [arXiv:2605.27958](https://arxiv.org/abs/2605.27958) — four-hypothesis test and style-robustness analysis on Gemma 3.
- *Probing classifiers* — Alain & Bengio, 2017 — foundational linear-probe paper for representation analysis.
- *Discovering Latent Knowledge in Language Models Without Supervision* (CCS) — Burns et al., 2023 — unsupervised probe variant.
