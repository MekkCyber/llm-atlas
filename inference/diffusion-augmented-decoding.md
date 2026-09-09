# Diffusion-Augmented Decoding
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Keep a standard autoregressive LLM's next-token-prediction distribution, but *sample* multiple tokens in parallel using a small distilled discrete-diffusion side-model whose stationary distribution matches the AR one. Because the sampler is calibrated to the AR marginals, there is no draft-verify acceptance-rejection step — the speedup is claimed to be *lossless* rather than "close to lossless."

**Prereqs:** [../fundamentals/attention](../fundamentals/attention.md), [../pre-training/mtp](../pre-training/mtp.md)
**Related:** [README](./README.md)

---

## What it is

Standard autoregressive decoding produces one token per forward pass; the sequential dependency is a hard floor on latency for long generations (reasoning traces, agent trajectories). Existing accelerators trade off:

| Family | Mechanism | Cost |
| --- | --- | --- |
| Speculative decoding (Medusa, EAGLE) | Draft model proposes $k$ tokens; base verifies | Two models to maintain; verify-reject wastes work |
| Multi-token prediction (MTP) | Extra heads predict future tokens | Distribution drifts from base AR marginals |
| **Diffusion-augmented decoding** | **Distilled discrete-diffusion sampler** draws multiple tokens jointly from the AR distribution | Only the AR base model + a small diffusion side-model |

The AR model implicitly defines a joint distribution over all future tokens; diffusion-augmented decoding trains a discrete-diffusion side-model to match *that joint*, then samples from it directly.

## How it works

1. Take a trained AR LLM as the target distribution.
2. Distill a small **discrete-diffusion** side-model whose sampling procedure has the AR joint as its stationary distribution.
3. At inference, a **specialized sampler** draws $k$ tokens in parallel from the diffusion side-model, calibrated so that the drawn tokens are distributed as if they had been produced token-by-token by the AR base.
4. Emit the parallel-drawn tokens with no rollback / rejection.

## Why it matters

- **Lossless in a regime where draft-verify is only approximately so.** Speculative decoding's "lossless" guarantee is at the level of the base distribution, but the base is used as a rejection filter — bad drafts still cost forward passes. Diffusion-augmented decoding sidesteps the reject step entirely.
- **Single-model deployment.** No separate draft model to keep in sync with the base after fine-tuning.
- **Composes with existing serving stacks.** The interface is still "output the next $k$ tokens"; the diffusion side-model is a drop-in.

## Gotchas & tricks

- **The diffusion side-model must actually track the base.** Distillation quality caps the achievable parallelism; a mis-calibrated sampler recovers speed but drifts from the AR marginals — silently losing the "lossless" property.
- **Effective parallelism is content-dependent.** Highly-constrained continuations (e.g. code closing brackets) parallelize; open-ended prose does not.
- **Not the same as diffusion LLMs.** Diffusion-augmented decoding keeps AR training and AR marginals; it just replaces the *sampling* loop with a diffusion procedure. Full diffusion LLMs replace training too.

## Sources

- Paper: *Unlocking Lossless Speedups in LLMs via Discrete Diffusion* — Sahoo et al., 2026 — [arXiv:2609.04010](https://arxiv.org/abs/2609.04010).
- Related: [../pre-training/mtp](../pre-training/mtp.md) — the training-time cousin (extra AR heads for future tokens); diffusion-augmented decoding is the sampling-time cousin.
