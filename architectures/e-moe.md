# E-MoE — Enhanced MoE for Non-Factorized Diffusion LMs
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A fix for masked diffusion LMs' few-step quality problem that uses the **MoE expert-routing decision as a discrete shared latent** instead of adding a continuous Gaussian latent. The reverse process becomes a mixture of factorised distributions, keyed by routing — correlated across positions, zero extra active parameters, no posterior-collapse risk.

**Prereqs:** [_moe.md](./_moe.md), [transformer-block.md](./transformer-block.md)
**Related:** [hc-dlm.md](./hc-dlm.md), [deepseek-moe.md](./deepseek-moe.md)

---

## What it is

Ivanov et al. (Skoltech / AIRI, 2026) target the known weakness of masked diffusion models (MDMs): the reverse process factorises over token positions, so when several tokens are unmasked in one step they are sampled independently from their per-position marginals — breaking inter-token dependencies and hurting few-step sample quality. The usual fix (a continuous Gaussian latent trained as a VAE) suffers posterior collapse. E-MoE replaces the continuous latent with the **expert-routing decision** of an MoE backbone.

## How it works

- **Routing as latent.** For each denoising step, the MoE router selects experts for the sequence. Conditional on that routing, the per-position distributions factorise as in standard MDM — but the shared routing choice correlates them.
- **Mixture of factorised distributions.** The full reverse-process distribution at a step is a mixture indexed by routing configurations: $p(x) = \sum_r p(r) \prod_i p(x_i \mid r)$. Each component is still a cheap factorised distribution; the mixture captures the correlation.
- **No extra active parameters.** MoE's sparse activation already determines per-token forward cost; using the routing as latent reuses the computation.
- **No posterior collapse.** Unlike VAEs, routing has no reconstruction decoder that can learn to ignore it — the routing *is* the forward pass, so it cannot be bypassed.

## Why it matters

- Delivers the correlation that continuous-latent diffusion LMs chase, without their training-time pathology.
- Few-step generation is the regime where diffusion LMs can beat autoregressive decoding on latency — E-MoE's correlated reverse process directly addresses the main quality cost of parallel decoding.
- Pairs naturally with [HC-DLM](./hc-dlm.md) — HC-DLM injects continuous structure, E-MoE injects discrete structure; the two are orthogonal.
- Improves few-step generation over factorised baselines on synthetic multi-modal benchmarks, binarized MNIST, and LM1B.

## Gotchas & tricks

- Latent capacity is bounded by the number of possible routing configurations — for small MoE top-k, this is modest; for larger top-k and more experts, the latent space is richer but routing becomes harder to learn.
- The method inherits MoE training pathologies (load balance, dead experts) — see [_moe.md](./_moe.md) and [aux-loss-free-balancing.md](./aux-loss-free-balancing.md).
- "Mixture of factorised distributions" is a theoretical framing; at generation time only one routing config is sampled per step, so the mixture structure is implicit.
- Scaling behavior from the LM1B-scale experiments to full-scale LM pretraining is open.

## Sources

- Paper: *E-MoE: Enhanced Mixture-of-Experts for Non-Factorized Diffusion Language Models* — Ivanov, Kolesov, Korotin, Oseledets, Goncharov; Skoltech / AIRI, 2026 — [arXiv:2609.37533](https://arxiv.org/abs/2609.37533).
- Context: *Masked Diffusion Models* line — Lou et al., 2023; Shi et al., 2024 — the factorised baseline this paper improves on.
- Context: *Continuous-latent diffusion LMs* — the posterior-collapse-prone family E-MoE sidesteps.
