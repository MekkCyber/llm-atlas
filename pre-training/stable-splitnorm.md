# Stable SplitNorm
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A normalization variant designed to **preserve a wide stability margin between the safe and optimal learning rate** during large-scale pre-training. In practice: lets a frontier recipe pick an LR close to the divergence boundary without divergence, which is where throughput-per-wall-clock peaks. Introduced in the IQuest-Q1 recipe alongside Muon and µP, as the stability leg of a three-leg pre-training stack (IQuest, 2026).

**Prereqs:** [_training-stability.md](_training-stability.md), [../fundamentals/attention.md](../fundamentals/attention.md)
**Related:** [../architectures/_normalization.md](../architectures/_normalization.md), [wsd-schedule.md](wsd-schedule.md), [_lr-schedules.md](_lr-schedules.md)

---

## What it is

A variant of RMSNorm (or LayerNorm, depending on application point) with a **split** structure — normalization statistics are computed over partitioned slices of the feature dimension rather than the full vector, and the per-slice scales are combined with a stability-preserving reparameterization. The design goal is explicitly stated: *preserve a wide stability margin between the safe LR and the optimal LR* across the pre-training run.

## How it works

- **Split structure.** The hidden vector is partitioned into groups; normalization statistics (variance, scale) are computed per group rather than across the whole vector. This limits the dynamic range any single feature can push the normalization layer into.
- **Stability-preserving scale.** The per-group affine (gain) is parameterised so that the gradient norm through the normalization layer is better-behaved under aggressive learning rates. The practical consequence is that the LR at which training first *diverges* is pushed well above the LR at which training is most *efficient*.
- **Placement.** Applied at the same positions as standard pre-norm (before attention and before FFN in each transformer block). The paper's recipe uses it uniformly across layers.
- **Composable with Muon and µP.** The three together form IQuest-Q1's pre-training stability leg: Stable SplitNorm handles activation-side stability, Muon handles optimizer-side stability for weight matrices, µP handles the parameterisation so that tuned hyperparameters transfer across widths.

## Why it matters

- **Wider stability margin = more aggressive LR = better throughput.** The gap between "safe LR" and "optimal LR" is the budget a frontier run has for aggressive scheduling. A wider gap means the recipe can push closer to optimal without divergence insurance.
- **Robustness across data-mix changes.** IQuest-Q1's Stage 1b shifts the code share from 40% to 70% in four phases. Each shift is a potential destabiliser; a wider margin absorbs the shift without an LR drop.
- **Composable with existing stability tricks.** Not a replacement for QKNorm, z-loss, or careful init — a layer on top of them, operating on the normalization stats themselves.

## Gotchas & tricks

- **Group count is a hyperparameter.** Too few groups ≈ standard norm (no benefit); too many groups ≈ per-feature norm (defeats the purpose). The paper's exact group count is not detailed in the abstract.
- **Can mask weak-signal failures.** A wider stability margin lets you run with an LR that would otherwise warn you (via divergence) that something upstream is wrong (data loader, bad init, misconfigured optimizer). Teams used to divergence as a canary should rely more on gradient-norm telemetry.
- **Checkpoint-compat concerns.** A Stable-SplitNorm-trained checkpoint cannot be loaded into a plain RMSNorm serving stack without the matching normalization code.
- **Not a free efficiency win.** The compute cost is comparable to RMSNorm; the win is in training stability, not throughput per step.

## Sources

- Paper: *IQuest-Q1: Advancing Foundation Capabilities for Agentic CLI Systems* — IQuest, 2026 — [arXiv:2610.30949](https://arxiv.org/abs/2610.30949) — introduces Stable SplitNorm as part of the stability-first pre-training stack.
