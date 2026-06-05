# Maximal Update Parametrization (μP)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** μP is a coordinated rescaling of initialization variances and per-layer learning rates such that *optimal hyperparameters transfer zero-shot across model widths*. You tune a small model, then scale to a large model under μP, and the same learning rate, init scale, and weight decay remain near-optimal. The huge practical payoff: no LR sweep at production scale. Originally derived for standard Transformers by Yang et al.; 2026 work extends μP to Gated Delta Networks and other linear-attention recurrents.

**Prereqs:** [_training-stability.md](_training-stability.md), [_lr-schedules.md](_lr-schedules.md)
**Related:** [fp8-training.md](fp8-training.md)

---

## What it is

Standard parametrization (SP) — the default Kaiming/Xavier-style init paired with one global learning rate — *does not* preserve the scale of activations and updates as a model's width grows. As you widen, gradient magnitudes shift in different ways per layer, and the LR that was optimal at small scale is wrong at large scale.

μP fixes this by setting per-layer init variances and per-layer LR multipliers so that, in the infinite-width limit and at finite width, every quantity that matters (forward-pass activation norms, gradient norms, parameter updates) has $\Theta(1)$ coordinate-wise magnitude regardless of width. Concretely:

| Layer kind | Init variance | LR multiplier (Adam) |
| --- | --- | --- |
| Input embedding | $\Theta(1)$ | $\Theta(1)$ |
| Hidden ($n \times n$) | $\Theta(1/n)$ | $\Theta(1/n)$ |
| Output (unembedding) | $\Theta(1/n)$ | $\Theta(1)$ |

This *coordinate-check* is the diagnostic: compute the coordinate-wise norm of every layer's activation and update at two widths, and confirm both are $\Theta(1)$. Any deviation is a missed scaling factor.

## How it works

The derivation is a careful propagation of coordinate-wise magnitudes through the forward pass, the backward pass, and the optimizer update. For a standard Transformer block, applying the rules above suffices. For non-standard blocks (gating, recurrent state, mixture-of-experts routing) you re-derive the rules block by block, ensuring the chain of $\Theta(1)$-magnitude inputs and outputs is preserved.

The 2026 extension to **Gated Delta Networks** is exactly this exercise: propagate magnitudes through the delta-rule recurrent state update and the output gate. The resulting parametrization is non-trivial — the standard Transformer μP rules do *not* apply directly because the recurrent state introduces a new layer kind.

## Why it matters

- **HP transfer.** Tune at e.g. 100M parameters, scale to 70B without re-sweeping. This was the bottleneck for every frontier training run before μP.
- **Stability at scale.** Loss spikes, dead neurons, divergent layers — many such failures trace back to mis-scaled init or LR. μP removes a large class of these.
- **Enables sub-quadratic scaling.** The μP-for-GDN work makes Gated Delta Network practically trainable at scale; previously every width required its own HP search, which was prohibitive past 1B params.

## Gotchas & tricks

- **Optimizer-specific.** The LR multiplier table differs for SGD, Adam, AdamW, and Adafactor. Use the rules derived for your optimizer.
- **Coordinate-check before scaling.** Always verify $\Theta(1)$ coordinate norms at two widths before committing. Missing one scaling factor is the most common bug.
- **Weight decay.** Decoupled weight decay (AdamW) interacts with μP — the decay strength should also be set per-layer in some derivations.
- **Doesn't replace the LR schedule.** μP transfers the *peak* LR. WSD, cosine, and warmup still apply on top.

## Sources

- Paper: *Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer* — Yang et al., 2022 — the original μP derivation for Transformers.
- Paper: *Unlocking Feature Learning in Gated Delta Networks at Scale* — Liu, Gu, 2026 — [arXiv:2606.04048](https://arxiv.org/abs/2606.04048) — extends μP to GDN.
- Related: Tensor Programs series for the underlying limit theory.
