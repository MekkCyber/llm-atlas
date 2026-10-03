# MoE Routing as a Shared Latent (E-MoE)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** In a masked diffusion LM, the reverse-process factorization severs correlations between positions decoded in the same step. E-MoE repurposes the Mixture-of-Experts router as a *shared discrete latent*: all positions in a sampling step condition on the same expert-routing choice, so the full reverse distribution becomes a mixture of factorized distributions. Captures inter-position correlations without extra parameters or an auxiliary VAE latent, and sidesteps VAE posterior-collapse.

**Prereqs:** [_moe.md](_moe.md), [transformer-block.md](transformer-block.md)
**Related:** [diffusion-lm.md](diffusion-lm.md) · [deepseek-moe.md](deepseek-moe.md) · [aux-loss-free-balancing.md](aux-loss-free-balancing.md)

---

## What it is

Masked diffusion LMs sample from $p(x_i | x_{\text{context}})$ independently per unmasked position, so a step that unmasks several positions misses the joint structure. Prior work added a continuous VAE latent $z$ to carry cross-position information; posterior collapse is a chronic failure mode. E-MoE's observation: an MoE's routing *is* a shared discrete latent. The router already picks the same expert indices for a sampling step, and if the model is conditioned on those indices, each position's conditional distribution carries that shared structure. Marginalizing over routing choices gives a mixture of factorized distributions over the discrete latent — hence "enhanced" vs the plain factorized baseline.

## How it works

- Backbone is a transformer with an MoE FFN block (standard top-$K$ routing).
- The reverse process writes:
  
  $p_\theta(x | x^k) = \sum_z \pi(z | x^k) \prod_i p_\theta(x_i | x^k, z)$
  
  where $z$ is the discrete routing decision (per layer / per step) and the inner product is factorized conditional on $z$.
- Training: standard masked-diffusion loss on $\log p_\theta(x | x^k)$, treating $z$ as a latent to marginalize via the router's own distribution.
- No auxiliary loss and no auxiliary VAE: the correlation mechanism is parameter-free beyond the MoE that's already there.

Because the router's output serves as the latent, active parameters per token are *the same* as a plain factorized MoE baseline.

## Why it matters

- **Fixes MDM few-step quality for free.** Standard MDMs trade speed for quality in low-step regimes; E-MoE recovers quality at the step counts where diffusion's wall-clock win matters.
- **Avoids VAE collapse.** Discrete routing can't silently become uninformative in the way a continuous latent can — the router's output is non-trivially non-uniform or load-balancing fails loudly.
- **Reuses standard infrastructure.** Any MoE codebase with top-$K$ routing can host E-MoE; no new scheduler, no new loss weights to tune.

## Gotchas & tricks

- **Benefits concentrate at low step counts.** At $K$ matching autoregressive steps, factorized baselines close the gap; E-MoE's win is in the regime where diffusion is actually faster than AR.
- **Load-balancing interacts with the latent.** If the aux-loss or aux-loss-free bias pushes the router toward uniform routing, the shared-latent informativeness drops; coordinated tuning needed. See [aux-loss-free-balancing.md](aux-loss-free-balancing.md).
- **Measured on synthetic/benchmark data (binarized MNIST, LM1B, synthetic multi-modal).** Scaling behavior on web-scale text is an open question as of late 2026.
- **Not a swap-in for autoregressive MoE routers.** The latent interpretation is specific to the diffusion reverse process; using the same router in an AR LM gives no coupling benefit.

## Sources

- Paper: *E-MoE: Enhanced Mixture-of-Experts for Non-Factorized Diffusion Language Models* — Ivanov, Kolesov, Korotin, Oseledets, Goncharov; Skoltech / AIRI, 2026 — [arXiv:2609.37533](https://arxiv.org/abs/2609.37533).
- Background: Masked diffusion LMs (D3PM, SEDD, MDLM) and the factorized reverse-process limitation.
