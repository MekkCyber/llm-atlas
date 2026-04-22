# Model Souping
*Depth — average the weights of several trained models into one final model.*

**TL;DR:** Train N models (different seeds, data orders, or fine-tuning recipes), then average their weights element-wise. The result often matches or beats any single model on validation loss and downstream benchmarks, for the cost of one extra optimizer call. Works because nearby minima in the loss landscape are **linearly connected** — a straight line between two trained checkpoints stays in low loss.

**Prereqs:** [transformer-block](../architectures/transformer-block.md)
**Related:** [mid-training](mid-training.md), [_lr-schedules](_lr-schedules.md), [olmo-2 case study](../case-studies/olmo-2.md)

---

## What it is

Given N trained checkpoints `{θ₁, θ₂, ..., θ_N}` that share the same architecture and were initialized identically (or fine-tuned from a common parent), produce a single final model by parameter-wise averaging:

```
θ_soup = (1/N) Σ_i θ_i
```

No retraining, no gradient steps — just one loop over the state dicts. The result is loaded and served as the final model.

The technique has two common flavors:

- **Uniform soup.** Average every candidate equally. Simplest, often strong.
- **Greedy soup.** Add candidates one at a time, keeping each only if it improves held-out accuracy. Strictly ≥ best single model by construction, usually close to uniform soup in practice.

## How it works

### Why averaging works — loss-landscape geometry

Frankle et al. (2020) empirically showed **linear mode connectivity**: two checkpoints `θ_A` and `θ_B` trained from the same initialization (or a shared early-training state) with different data orderings tend to satisfy the property that the loss along the segment `(1-t) θ_A + t θ_B` stays low for all `t ∈ [0, 1]`. They are in the same basin. Models trained from **different** random initializations typically are not — averaging them lands on a loss barrier well above either endpoint.

If the basin is approximately quadratic near its floor, the midpoint has *lower* loss than either endpoint (the quadratic's minimum lives at the basin floor, and both endpoints are near-floor but slightly off in different directions). Averaging many endpoints = concentrating mass near the basin floor.

The original Model Soups paper (Wortsman et al., 2022) used shared-parent candidates throughout — all fine-tuned from the same pretrained CLIP — so its results do not directly speak to cross-init averaging.

### What varies across candidates

Useful sources of diversity, in roughly decreasing impact:

1. **Data order / seed.** Same recipe, different shuffles. Cheap, surprisingly effective.
2. **Hyperparameter sweeps.** Slightly different LR, weight decay, or warmup schedules.
3. **Fine-tuning mix variations.** E.g., multiple SFT runs with different data ratios.
4. **Checkpoints from different steps** of the same run — averaging the last K checkpoints ("Stochastic Weight Averaging / EMA") is the degenerate one-run version.

### OLMo 2's application

OLMo 2 averages Stage-2 runs that share the Stage-1 base model but used different Stage-2 data orderings and seeds. The souped model is the final release. Measured lift: ~0.3–1.0 points across benchmarks, essentially free given the runs existed anyway.

Llama 3 reports similar practice, averaging final SFT candidates. The technique is now standard in large-lab releases.

## Why it matters

- **Free quality.** The marginal cost is negligible compared to training any one candidate.
- **Stabilizes evaluation.** Single-run variance on benchmarks is real (±0.5–1.0 points). Souping damps that variance toward the mean of a nearby valley.
- **Natural fit for large releases.** Labs already run multiple candidates for safety/ablation. Souping is a way to not throw the also-rans away.

## Gotchas & tricks

- **Shared parent is required for naive averaging.** Naive element-wise averaging assumes the candidates lie in the same loss basin. Empirically this holds when they share an initialization or an early-training checkpoint, and fails across different random inits (Frankle et al. 2020). Permutation-aligned merging (**Git Re-Basin**, Ainsworth et al. 2022; **OT Fusion**, Singh & Jaggi 2020) can extend averaging across different inits by aligning neurons before averaging — but that is no longer "soup" in the Wortsman sense, and is rarely used at LLM scale.
- **Equal architecture only.** Different vocab sizes, different head counts — no soup.
- **Optimizer state is not averaged.** You soup the model weights. The optimizer state (Adam m, v) is discarded — souping produces a finished model, not a training-resume point.
- **Don't soup across very different recipes.** Averaging a SFT'd model with a DPO'd model can work but requires care — they've moved in different directions from the shared parent. Safer to soup multiple SFT runs or multiple DPO runs, not cross-recipe.
- **Check each candidate is actually competitive.** A soup of one good and one broken model is worse than the good one. Greedy souping handles this automatically; uniform souping doesn't.

## Sources

- Paper: *Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time* — Wortsman et al., 2022 — introduces the term and greedy vs. uniform variants.
- Paper: *SWA: Averaging Weights Leads to Wider Optima and Better Generalization* — Izmailov et al., 2018 — the single-run precursor.
- Paper: *2 OLMo 2 Furious* — AI2, 2024 — applies souping to Stage-2 runs.
- Paper: *The Llama 3 Herd of Models* — Meta, 2024 — applies souping to SFT candidates.
