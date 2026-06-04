# Deception Probes

*Depth — linear probes trained on LLM activations to score whether the model's generation is deceptive.*

**TL;DR:** A deception probe is a linear classifier trained on LLM activations to flag deceptive outputs without ground-truth labels. Probes are proposed as cheap automated monitors for safety. Clean-benchmark AUROC is near-perfect (>0.998), but the metric collapses under distributional shift — and the deception representation is multi-dimensional, not a single direction. Style augmentation during probe training mostly rescues robustness; the conceptual question of *what* the probe is detecting (a single deception feature vs an entropy proxy vs a stylistic confound) remains open.

**Prereqs:** [linear-probes.md](./linear-probes.md), [../safety/deceptive-alignment.md](../safety/deceptive-alignment.md)
**Related:** [../safety/scheming.md](../safety/scheming.md), [../safety/cot-monitoring.md](../safety/cot-monitoring.md), [../safety/alignment-faking.md](../safety/alignment-faking.md)

---

## What it is

A deception probe is the safety-evaluation specialization of [linear probes](./linear-probes.md): activations from a frozen LLM are scored by a linear classifier trained on examples of *deceptive* vs *honest* generations. The deployment story is "stick the probe on the model's residual stream, get a deception score per generation, gate / flag accordingly." That story has been the basis for several proposed monitor pipelines.

The pressure-test paper (Kumar 2026, Gemma 3 1B–27B) interrogates the story along four axes:

1. Is deception a single linear direction?
2. Or a multi-dim subspace?
3. Or a convex conic hull (deception is a cone of related directions)?
4. Or just a proxy for computational entropy / model confusion?

## How it works

Standard pipeline:

```
prompts_honest = {q : q has honest-completion exemplar}
prompts_deceptive = {q : q has deceptive-completion exemplar}
acts_honest = activations(model, prompts_honest, layer=ℓ, position=end-of-assistant)
acts_deceptive = activations(model, prompts_deceptive, layer=ℓ, position=end-of-assistant)
w, b = logistic_regression(stack(acts_honest, acts_deceptive), labels)
deception_score(generation) = σ(w · activations(model, generation, layer=ℓ, position=end))
```

Pressure-test additions:

- **Cross-domain transfer matrix.** Train on domain A's honest/deceptive pairs, test on domain B's. Off-diagonal collapse means the probe fit domain features, not deception.
- **Multi-dim ($k > 1$) probe.** Fit a $k$-dim projection; test whether AUROC saturates as $k$ grows.
- **Permutation null.** Refit on shuffled labels to subtract chance-level baseline.
- **Entropy residualization.** Regress activation norm / attention entropy out first; tests the "deception = confusion" hypothesis.
- **Stylistic distractors.** Evaluate on the same target tasks reworded in 8 distinct writing styles.

## Why it matters

- **Cheap deployable monitor — *if* robust.** Probes cost ~zero compute compared to chain-of-thought monitoring or a separate judge model. If they were robust, they'd be the obvious monitor channel.
- **Robustness is the whole story.** Without style augmentation, deception probes collapse under deployment distribution shift, while reporting near-perfect AUROC on clean data. A monitor that silently fails on the very distribution shift it's supposed to catch is worse than no monitor.
- **Single-direction hypothesis is dead.** $k=1$ captures only 0.61–0.80 of the signal across Gemma 3 scales. Multi-dim training data and multi-dim probes are non-negotiable.

## Gotchas & tricks

- **Always evaluate cross-domain.** Within-distribution AUROC is meaningless for deception monitoring. Cross-domain transfer is the realistic metric.
- **Style augmentation works.** Training the probe with style-varied prompts (formal / casual / poetic / technical) recovers mean AUROC 0.979–0.983 on unseen styles. Cheap defense; should be standard.
- **Entropy residualization changes the verdict.** Some "deception" probes lose most of their signal once activation norm / attention entropy is regressed out — they were just confidence proxies.
- **Layer choice is brittle.** Probe quality often peaks at a specific layer (often two-thirds depth). The peak shifts across model sizes; sweep don't guess.
- **Probes ≠ causal evidence.** A deception probe is a detector, not an explanation. Activation patching is needed to claim the model *uses* the direction.

## Sources

- *Pressure-Testing Deception Probes in LLMs: Scaling, Robustness, and the Geometry of Deceptive Representations* — Kumar, 2026 — [arXiv:2605.27958](https://arxiv.org/abs/2605.27958) — primary source. Gemma 3 1B–27B sweep, four-hypothesis test, style-augmentation defense.
- *Representation Engineering* — Zou et al., 2023 — earlier probe-based deception/honesty detection.
- *Alignment Faking in Large Language Models* — Anthropic, 2024 — provides the deception examples many probe pipelines train on.
