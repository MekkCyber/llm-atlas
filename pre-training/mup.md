# Maximal Update Parameterization (µP)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A parameterization of layer widths, initializations, learning rates, and (for some variants) optimizer scalings such that the **optimal learning rate is invariant to model width**. Tune the LR on a small proxy width; transfer directly to the target width. Introduced by Yang & Hu (2021) as the "Tensor Programs V / µTransfer" line; now the standard machinery for setting LR on frontier LLMs, with modern extensions covering MoE, Multi-head Latent Attention, and the Muon optimizer.

**Prereqs:** [_lr-schedules.md](./_lr-schedules.md), [_training-stability.md](./_training-stability.md)
**Related:** [muon-optimizer.md](./muon-optimizer.md) · [../architectures/_moe.md](../architectures/_moe.md) · [../architectures/mla.md](../architectures/mla.md)

---

## What it is

Standard (SP, "standard parameterization") transformers behave differently at different widths — the optimal LR shifts with width, so tuning at 100M and copying to 100B doesn't work. µP fixes the shift by scaling three families of quantities so that the **feature updates** (the change in a hidden state during training) stay `Θ(1)` in width. LR that works at width 128 works at width 8192.

## How it works

µP prescribes three width-dependent scalings, with `d` the width and `d_proxy` a reference:

```
Init variance   ×  factor(d)        # depends on tensor type: input vs hidden vs output
Forward output  ×  factor(d)        # applied to the readout / output layer
Learning rate   ×  factor(d)        # per parameter group
```

Different tensor groups (embedding, hidden weights, output weights, biases, gains) get different scalings. The bookkeeping is captured in tables such as the *µParam Table* in Yang & Hu (2021); modern implementations expose it as a library (`mup` package, or hand-rolled per-model).

For AdamW-based training, the LR scaling is `Θ(1/d)` on hidden weights. For **Muon** (matrix-sign optimizer) the update is already unit-singular-value, so the width-dependence of the effective update magnitude changes — a Muon-parameterized model transfers optimal LR across widths *without* the explicit LR/`d` factor, provided the initialization and forward scalings still follow µP.

## Why it matters

- **Kills the biggest cost item in HP tuning at scale.** Sweeping LR at 100B is prohibitive; sweeping at 500M and transferring costs ~1000× less. Every modern lab does this either by µP or ad-hoc scaling laws.
- **Transfer across widths, not just LR.** Optimal `batch_size`, `warmup`, and `β1`/`β2` also transfer under µP with lower variance than under SP.
- **Composes with token-axis scaling.** µP transfers across widths at fixed tokens. Kim et al. (2026) show you can extend to trillion-token horizons by fitting a linear regression of optimal-LR vs `log(tokens)` on top of µP width-transfer — R²=0.95 extrapolation from small proxies to a 155B/17B MoE.
- **Works for modern MoE + MLA.** The classical µP tables were derived for dense transformers. Kim et al. adapt the derivation for routed experts and Multi-head Latent Attention, keeping width transfer under the Muon optimizer.

## Gotchas & tricks

- **Not automatic.** Every new architecture element (attention variant, routing, normalization) needs its µP scaling worked out. Off-the-shelf µP libraries assume a standard transformer.
- **Multiple valid µP tables exist.** The "canonical" µP (Yang & Hu 2021) is one point; there are variants (SP-µP, µParam-2, spectral µP). Confirm which one your library implements.
- **Depth transfer is a separate problem.** Classical µP transfers across **width** at fixed depth. Depth transfer needs additional care (see Yang et al. 2023 "Tensor Programs VI" or the depth-µP extensions).
- **Loss curves at proxy width are noisy.** Fit optimal LR on the *lowest smoothed loss over a small LR grid*, not the single best noisy point. Otherwise you extrapolate a lucky seed.
- **Only holds if numerics are healthy.** µP assumes forward and backward passes are stable at all widths. FP8 training, gradient clipping, and QK-Norm interact with the assumptions — sanity-check the transfer with a spot run at intermediate width.

## Sources

- Paper: *Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer* — Yang & Hu, 2021 — the µP formulation and µTransfer procedure.
- Paper: *Tensor Programs VI* — Yang et al., 2023 — depth extensions.
- Paper: *Let's Scale Step by Step: Compute-Efficient Hyperparameter Transfer for Large-Scale Mixture-of-Experts* — Kim et al., 2026 — µP + Muon + MLA + MoE and the token-axis scaling law.
- Code: `mup` package — https://github.com/microsoft/mup
