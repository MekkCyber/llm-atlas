# DepthBench
*Depth — a controlled benchmark that varies the width/depth aspect ratio of Transformer stacks at fixed parameter count to measure how much depth actually helps.*

**TL;DR:** DepthBench holds parameter count constant while sweeping width vs depth across 10 residual/normalization variants. Result: whether depth functions as a real scaling axis depends **critically** on the residual connection design; standard Pre-LN often *degrades* in deep-and-narrow regimes, while HighwayC and Full AttnRes keep gaining. The benchmark gives a comparable "effective computational depth" number across architectures.

**Prereqs:** [../architectures/transformer-block.md](../architectures/transformer-block.md), [../architectures/_normalization.md](../architectures/_normalization.md)
**Related:** [../architectures/reordered-norm.md](../architectures/reordered-norm.md), [../pre-training/_training-stability.md](../pre-training/_training-stability.md)

---

## What it is

An evaluation harness for the question "does adding depth actually help this architecture?" — isolated from confounds like parameter count, data mix, and optimizer choice. It benchmarks 10 residual/normalization variants across a matched sweep of width-to-depth ratios and reports whether accuracy improves, plateaus, or regresses as depth grows.

## How it works

- **Matched sweeps.** For each architecture, models are trained at several aspect ratios (deep-and-narrow vs shallow-and-wide) with identical parameter budgets, identical data, and identical optimizer settings. This removes the usual confounds when comparing depth across papers.
- **10 variants.** Standard Pre-LN, Post-LN, ReZero, DeepNorm, HighwayC, Full AttnRes, and several ablations of residual gating and normalization placement.
- **Metric.** "Effective computational depth" — a scalar summarizing how far along the depth axis the architecture continues to gain, before plateauing or degrading.
- **Diagnostics.** Per-layer activation and gradient norm plots let readers see *where* deep Pre-LN loses signal — usually the last few blocks contribute negligibly.

## Why it matters

Depth is the frontier scaling axis for reasoning-focused models: deep-and-narrow stacks (e.g. 128×small vs 32×big) have been reported to improve reasoning quality at fixed parameter count. But if the residual design silently caps effective depth, the community may be under-selling the scaling curve. DepthBench turns "which residual do I pick?" from a folklore question into a benchmarked one, with concrete guidance on architectures to prefer at large aspect ratios (HighwayC, Full AttnRes) vs those to avoid (unmodified Pre-LN in deep-narrow regimes).

## Gotchas & tricks

- **Optimizer coupling.** Some residual variants (DeepNorm, HighwayC) require adjusted init scales; use the paper's recommended settings, otherwise gains from the architecture leak into optimizer noise.
- **Small-model artifacts.** Very small runs (≪ 100M params) show noisier aspect-ratio effects; DepthBench's headline numbers are at model sizes where the ordering has stabilized.
- **Data budget matters.** At severe under-training, all variants look similar; DepthBench's evaluation regime uses roughly Chinchilla-optimal or slightly over-trained data budgets.
- Complements — does not replace — full downstream evaluations; a small effective-depth win doesn't always translate to reasoning-benchmark wins.

## Sources

- Paper: DepthBench: Measuring How Residual Connections Enable More Computational Depth — Wang, Huang, Kang, González-Martínez, Liu, Liu, 2026 — [arXiv:2609.32534](https://arxiv.org/abs/2609.32534)
