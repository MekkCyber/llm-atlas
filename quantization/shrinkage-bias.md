# Shrinkage Bias
*Depth — the systematic negative rounding error inherent in non-uniform low-precision grids.*

**TL;DR:** Non-uniform 4-bit floating-point formats (E2M1, FP4) have **logarithmically spaced bins** — narrow near zero, wide at the high end. Rounding values inside a wide bin (whether deterministic round-to-nearest or stochastic) systematically biases them toward zero, producing a small negative drift on every operation. Across ~100 transformer layers, this accumulates multiplicatively into measurable BF16-relative loss degradation. Random Hadamard Transform amplifies (not cancels) the bias because it makes more values land in the wide outer bins. **Uniform grids (E1M2, INT4) avoid the bias entirely.** Diagnosed in the UFP4 paper, Ant Group Ling Team, arXiv 2606.20381.

**Prereqs:** [_number-formats](_number-formats.md)
**Related:** [fp4-training](fp4-training.md), [fp8](fp8.md), [../pre-training/_training-stability.md](../pre-training/_training-stability.md)

---

## What it is

A geometric-origin rounding error that affects any low-precision floating-point format whose representable values are not uniformly spaced. The bias is **negative** (rounds toward zero), **systematic** (every rounding event contributes the same direction), and **multiplicative across layers**.

For 4-bit FP formats the effect is large enough to dominate loss-degradation budgets in pretraining. For higher-precision formats (FP8, BF16) the bias exists but is small enough to ignore.

## How it works

### Why non-uniform grids bias toward zero

A non-uniform FP4 format (e.g. E2M1) has 16 representable values spaced like:

```
... -6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0,
    0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0
```

Bin widths near zero are 0.5; bin widths near 6.0 are 2.0.

Consider a continuous input value `x = 5.0`, lying in the bin `[4.0, 6.0]`:
- **Round-to-nearest** → 4.0 (distance 1.0) or 6.0 (distance 1.0); ties handled by RNE.
- **Stochastic rounding** → 4.0 with probability $(6-5)/(6-4) = 0.5$, 6.0 with probability 0.5.

So far so good for `x = 5.0` — symmetric around the bin midpoint.

But consider `x = 4.5`:
- **Stochastic rounding** → 4.0 with prob $(6-4.5)/(6-4) = 0.75$, 6.0 with prob 0.25.
- **Expected value** = $0.75 \cdot 4.0 + 0.25 \cdot 6.0 = 4.5$ ✓ (unbiased per element).

So **per-element stochastic rounding is unbiased**. Where does the bias come from?

### The accumulation argument

Per element it's unbiased; per GEMM it's biased. The reason is the **distribution of values** entering the GEMM after Random Hadamard Transform. RHT redistributes mass to make use of more of the dynamic range, which pushes values *into* the wide outer bins — exactly the regime where the per-element rounding is least precise. Over a long inner-dim sum, the accumulator picks up the variance from those wide bins as systematic noise that compounds.

In a uniform-grid format (E1M2 or INT4), every bin has the same width, so RHT's redistribution doesn't change the per-element rounding variance — uniform grids "convert the improved bucket utilization from RHT into higher quantization quality" (paper's framing).

### Why the bias compounds across layers

If each layer's matmul introduces a small negative bias $\delta$ on its output activations, by layer $L$ the cumulative bias scales as $L \cdot \delta$ on the loss landscape. The paper shows this is the unified explanation for the training instability observed in published E2M1 recipes — it's not a recipe artifact, it's the format.

## Why it matters

- **Falsifies "FP4 just needs better recipes" narratives.** No scaling trick fixes a grid-geometry bias.
- **Direct accelerator-design consequence.** Hardware roadmaps centered on E2M1 are betting on the wrong geometry; the paper recommends uniform 4-bit grids as first-class training primitives.
- **Generalizes the diagnostic frame.** Any future low-precision format with non-uniform bins inherits this bias; check the grid geometry before committing to a hardware path.

## Gotchas & tricks

- **Per-element unbiasedness ≠ accumulator unbiasedness** under realistic input distributions. Be careful not to argue "stochastic rounding is unbiased per element, so the GEMM is unbiased."
- **RHT amplifies, not cancels, the bias** in non-uniform grids — counterintuitive, since RHT is usually pitched as outlier-handling.
- **Easy to miss in small-scale ablations.** A 1B-parameter run with 50 layers shows mild bias; a 100B run with 60+ layers shows the failure mode clearly. Validate at scale.
- **Per-layer compensation doesn't work** at scale. Adding a per-layer learned bias to correct for shrinkage doesn't generalize — the bias depends on the input distribution, which changes with the model.

## Sources

- Paper: *Rethinking Shrinkage Bias in LLM FP4 Pretraining: Geometric Origin, Systemic Impact, and UFP4 Recipe* — Chen, Tian, Jiang, Zhang, Yu, Jiang, Gong, Liu, Liu, Zhang, Zhou, Ant Group (Ling Team), 2026, arXiv 2606.20381.
- Spec: *OCP Microscaling Formats (MX) v1.0* — Open Compute Project, 2023 — E2M1 format definition.
