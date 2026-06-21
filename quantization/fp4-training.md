# FP4 Training
*Depth — end-to-end pretraining in 4-bit floating point, including UFP4 (uniform-grid) and E2M1 recipes.*

**TL;DR:** FP4 pretraining promises ~2× memory and throughput over FP8 — but the element-format choice is consequential. Current hardware (Blackwell, Rubin-class, MI350-series) centers on **E2M1**, whose non-uniform bin geometry suffers from **shrinkage bias** that compounds across layers. **UFP4** (Ant Group Ling Team, arXiv 2606.20381) replaces E2M1 with a uniform-grid 4-bit format (E1M2 or INT4), applies the Random Hadamard Transform to all three training GEMMs, and restricts stochastic rounding to the dY GEMM. Achieves consistently lower BF16-relative loss degradation than strong E2M1 baselines on Dense 1.5B, MoE 7.9B, and MoE 124B long-run pretraining.

**Prereqs:** [_number-formats](_number-formats.md), [fp8](fp8.md), [shrinkage-bias](shrinkage-bias.md)
**Related:** [../pre-training/fp8-training.md](../pre-training/fp8-training.md), [../pre-training/_training-stability.md](../pre-training/_training-stability.md)

---

## What it is

The recipe family for pretraining LLMs end-to-end in 4-bit floating-point GEMMs (forward, dY = ∂L/∂Y, dW = ∂L/∂W), with master weights and optimizer state at higher precision. Two distinct sub-families:

- **E2M1-based** (industry default 2024–2026): non-uniform 4-bit elements, MXFP4-style 32-element E8M0 block scales, RHT for outlier handling. Used by NVIDIA's Transformer Engine FP4 path and most published Blackwell FP4 work.
- **UFP4** (uniform-grid, 2026): replaces E2M1 with E1M2 or INT4 — uniformly spaced bins — keeps RHT on all three GEMMs, but **restricts stochastic rounding to the dY GEMM**.

## How it works

### The three training GEMMs

```
forward    : Y  = X @ W
backward-A : dX = dY @ Wᵀ
backward-W : dW = Xᵀ @ dY
```

In FP4 training each GEMM is FP4 × FP4 → BF16 accumulator. The interesting choices are (a) which element format to use for each operand, (b) what scale granularity, (c) whether to apply RHT to absorb outliers, (d) where to round stochastically.

### Why uniform grids win at 4 bits

The standard E2M1 format has 16 representable values, but they're spaced **logarithmically** — bins near zero are narrow, bins at the high end are wide. Under stochastic rounding (or even deterministic round-to-nearest), values near the edge of a wide bin are systematically rounded toward zero — a small negative bias on every operation. With ~100 layers, this compounds multiplicatively into measurable loss degradation.

Uniform grids (E1M2 with 1 exponent / 2 mantissa, or INT4) space all bins equally. No grid-geometry bias. See [shrinkage-bias](shrinkage-bias.md) for the full analysis.

### The role of RHT

Random Hadamard Transform applied to GEMM inputs spreads outlier mass across many channels — making the values more uniformly distributed and so making better use of the 16 bins. UFP4 applies RHT to all three training GEMMs (E2M1 recipes typically apply it selectively).

### Why stochastic rounding only on dY

UFP4's distinctive ablation: stochastic rounding everywhere actually *hurts* with uniform grids, because the unbiased rounding interacts badly with RHT's whitening. Restricting stochastic rounding to the gradient-w.r.t.-output (dY) GEMM — where the gradient distribution has the heaviest tails — keeps the variance reduction where it helps and avoids the harm elsewhere.

### What stays high-precision

Same template as FP8 training: master weights FP32, embedding/LM head BF16, normalization FP32, attention softmax BF16/FP32, MoE router BF16. Only the big matmuls are FP4.

## Why it matters

- **Hardware-roadmap implication.** Blackwell / Rubin / MI350 all bet on E2M1; UFP4 argues they should add E1M2/INT4 as first-class training primitives. Has direct accelerator-design consequences for the next generation.
- **Quality at scale.** UFP4's gap over E2M1 baselines *widens* with model scale (Dense 1.5B → MoE 7.9B → MoE 124B), supported by scaling-law fits.
- **~2× memory/throughput over FP8** when the recipe holds up — the cost-reduction lever for the next generation of frontier models.

## Gotchas & tricks

- **E2M1's shrinkage bias is grid-geometry, not recipe.** No amount of clever scaling fixes it; you need a uniform-grid element format.
- **RHT must be applied consistently across all training GEMMs** in UFP4. Selective RHT (e.g. forward only) leaves a recipe that mixes uniform-grid bias absence on some GEMMs with E2M1-style behavior on others.
- **Stochastic rounding placement matters.** Naively applying stochastic rounding everywhere with uniform grids hurts; restricting to dY captures the benefit without the cost.
- **Accumulation precision is still load-bearing.** Same as FP8: tensor-core-native accumulation over long inner dims loses precision; periodic FP32 promotion is still required.
- **Validate at scale.** As with FP8, small-model FP4 runs are stable; real failure modes show up at 100B+ scale. Don't draw conclusions from 1B ablations alone.

## Sources

- Paper: *Rethinking Shrinkage Bias in LLM FP4 Pretraining: Geometric Origin, Systemic Impact, and UFP4 Recipe* — Kunlong Chen, Changxin Tian, Zhonghui Jiang, Haitao Zhang, Chaofan Yu, Peijie Jiang, Mingliang Gong, Jia Liu, Ziqi Liu, Zhiqiang Zhang, Jun Zhou, Ant Group (Ling Team), 2026, arXiv 2606.20381.
- Spec: *OCP Microscaling Formats (MX) v1.0* — Open Compute Project, 2023 — MXFP4 and E8M0 scale.
- Paper: *Microscaling Data Formats for Deep Learning* — Rouhani et al., Microsoft, 2023 — the MX paper behind the OCP spec.
- NVIDIA Blackwell architecture whitepaper — hardware FP4 (E2M1) tensor cores.
