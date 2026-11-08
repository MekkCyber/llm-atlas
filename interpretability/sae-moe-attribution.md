# SAE-MoE expert attribution
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Train a sparse autoencoder on the **output** of a MoE layer. Because the MoE output is a router-weighted sum of expert outputs and the SAE encoder is affine, each SAE latent's activation decomposes **exactly** as a linear combination of per-expert contributions. On Qwen3-30B-A3B this reveals that MoE experts specialise at a far finer grain than router gating suggests.

**Prereqs:** [sparse-autoencoder](sparse-autoencoder.md), [_moe](../architectures/_moe.md)
**Related:** [deepseek-moe](../architectures/deepseek-moe.md), [aux-loss-free-balancing](../architectures/aux-loss-free-balancing.md)

---

## What it is

MoE interpretability has been bottlenecked by one question: *what does an expert actually do?* Looking at gating probabilities tells you *when* the router picks expert i, not *what* that expert contributes. SAE-MoE attribution answers the second question.

## How it works

Let the MoE layer output for token `x` be:

```
y = Σ_i g_i(x) · f_i(x)
```

where `g_i` is the router gate for expert `i` and `f_i` is expert `i`'s output. Train an SAE on `y` so that:

```
z_j = ReLU(w_j · y + b_j)
```

where `w_j` is the encoder row for latent `j`. Because `y` is a linear sum over experts and the encoder is affine:

```
z_j ≈ ReLU(Σ_i g_i(x) · (w_j · f_i(x)) + b_j)
```

The **per-expert contribution** to latent `j` on input `x` is `g_i(x) · (w_j · f_i(x))`. This is an *exact* decomposition (up to the ReLU) — no approximation.

Aggregated across many inputs, compute each latent's distribution over experts. If one expert dominates the contribution, that latent is a specialised expert's feature.

## The empirical finding

Applied to Qwen3-30B-A3B at layers 4, 30, 46, the paper shows:

- **Expert contribution is far more concentrated than gating.** The router may activate several experts with comparable probability, but one expert typically drives the *interpretable* features.
- **Fine-grained specialisation.** Auto-interp labels reveal experts carry sub-concepts — not just "code" or "math" but sub-domains like "Python exception messages", "LaTeX symbols", or "French verb tense agreement".
- **Layer differences.** Early-layer experts specialise on surface forms; late-layer experts carry more semantic / task-level roles.

## Why it matters

The MoE interpretability problem was open: routers and experts co-adapt in opaque ways, and raw gating stats are misleading. The SAE-MoE attribution decomposition is simple, exact, and gives a principled per-expert view. This matters for both scientific understanding (what *is* expert specialisation?) and practical work: expert pruning, expert-level safety filtering, and training interventions that target specialisation directly rather than only load balance.

## Gotchas & tricks

- **Train the SAE on the layer output**, not on the individual expert outputs, or the decomposition loses its exact-ness.
- **Router entropy matters.** Very high router entropy spreads contributions across experts and makes any single expert's share look small. Filter tokens by router confidence before aggregating.
- **Shared experts** (as in DeepSeek-MoE) appear in every decomposition and should be analysed separately from routed experts.

## Sources

- Paper: SAE Interprets MoE: Sparse AutoEncoders Reveal That MoE Experts Exhibit Human-Interpretable Fine-Grained Specialization — Zhuo et al., 2026 — [arXiv:2611.08089](https://arxiv.org/abs/2611.08089)
