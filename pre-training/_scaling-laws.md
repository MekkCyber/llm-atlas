# Scaling laws
*Taxonomy — fitted forms predicting training loss as a function of model and data size.*

**TL;DR:** A scaling law $L(N, D)$ predicts pretraining loss from parameter count $N$ and token count $D$. Successive generations have argued about whether $N$ and $D$ enter separately (Chinchilla) or coupled (Kaplan), because the answer decides where along the compute-optimal frontier to train. The 2026 Skaling law revives coupling with a single interaction exponent that fits both regimes better.

**Related taxonomies:** [_lr-schedules.md](_lr-schedules.md), [_training-stability.md](_training-stability.md)
**Depth files covered here:** [skaling-law.md](skaling-law.md) · [wsd-schedule.md](wsd-schedule.md)

---

## The problem

Every frontier pretraining run bets millions of GPU-hours on an extrapolation from smaller-scale sweeps. If the fitted law is wrong at the corners of $(N, D)$ space where you extrapolate, you train at the wrong point on the compute-optimal frontier — undertraining a model or wasting tokens.

## The shared pattern

All variants fit a functional form:

$$
L(N, D) \approx E + f(N) + g(D) + \text{(interaction term)}
$$

$E$ is the irreducible loss; $f, g$ are power-law falloffs in model size and data. The generations disagree on the interaction term. The fits are done on Huber-loss regression of measured pretraining losses across a grid of $(N, D)$.

## Variants

| Technique | Key idea | Main tradeoff | When it wins |
| --- | --- | --- | --- |
| Kaplan (2020) | $N$ and $D$ tightly coupled; loss depends on compute $C \approx 6ND$ | Under-predicts benefit of more tokens at fixed compute → recommends bigger models than optimal | Historical / establishes the framework |
| Chinchilla (2022) | Separable additive terms in $N$ and $D$; recommends ~20 tokens/param | Systematically wrong at overtraining ($D \gg N$) and data-scarce extremes | Modern default for "compute-optimal" planning |
| [skaling-law](skaling-law.md) | Chinchilla form + single interaction exponent coupling $N$ and $D$ | One extra parameter; still assumes power-law shape | Extreme regimes (heavy overtraining, data-scarce); sparse-grid extrapolation |
| Emergent-capability laws (Wei et al., no depth file yet) | Predict downstream benchmark scores instead of loss | Threshold behavior harder to fit; disputed by "mirage" analyses | Cross-benchmark capability planning |

## How to choose

The **modern default** for pretraining compute planning is Chinchilla — it's simple, well-fit in the moderate regime, and every open-model recipe references it. Use [skaling-law](skaling-law.md) when your target is far from Chinchilla-optimal (>10× overtraining, or unusually small $D$) or when you want to run a sparse-grid sweep instead of a full one.

Downstream-capability laws are useful for benchmark planning but should not be trusted for compute allocation — the loss laws are more empirically robust.

## Adjacent but distinct

- [_lr-schedules.md](_lr-schedules.md) — scheduling matters as much as $(N, D)$; scaling laws assume a well-tuned schedule underneath.
- [mid-training.md](mid-training.md) — mid-training extends the pretraining recipe; its compute isn't captured in the classical $C = 6ND$ accounting.

## Sources

- Paper: *Scaling Laws for Neural Language Models* — Kaplan et al., 2020.
- Paper: *Training Compute-Optimal Large Language Models* — Hoffmann et al., 2022 (Chinchilla).
- Paper: *Skaling: Chinchilla's Exponents Meet Kaplan's Coupling* — Videau et al., FAIR, 2026 — arXiv:2608.07222.
