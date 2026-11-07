# SAE-Based Probing of MoE Experts
*Depth — using SAE features to attribute what each MoE expert specializes in.*

**TL;DR:** Fit a **sparse autoencoder** on the pre-routing hidden states feeding an MoE layer, then for each SAE feature record the distribution of expert-routing probabilities on examples that fire the feature. A concentrated expert distribution means the feature is *owned* by a specific expert; aggregating across features gives a human-interpretable description of each expert's specialization. The method surfaces specializations invisible to coarse router / domain statistics.

**Prereqs:** [sae](./sae.md), [_moe](../architectures/_moe.md)
**Related:** [deepseek-moe](../architectures/deepseek-moe.md), [aux-loss-free-balancing](../architectures/aux-loss-free-balancing.md)

---

## What it is

MoE LLMs route each token to a small number of experts. The standard way to ask "what does expert *k* do?" is to look at the token-type / domain distribution of inputs it receives. That's coarse: experts route by latent features, not tokens or domains. SAE-based probing closes the gap by using SAE features (which *are* latent features) as the lens.

## How it works

Given an MoE layer with experts $\{E_1, \ldots, E_N\}$ and router $R(x) \to$ probability vector over experts:

1. Fit an SAE on the residual-stream hidden state $x$ *at the input of the MoE layer*. Follow standard [sae](./sae.md) training; top-K with large $m/d$.
2. For each SAE feature $i$, collect every example where $f_i > 0$ and record $R(x) = p^{(i)} \in \Delta^{N-1}$.
3. Define the **feature-expert attribution**: $P_i = \mathbb{E}[p^{(i)}]$ — the mean router distribution on examples that fire feature $i$.
4. Score concentration: entropy $H(P_i)$ or max-probability $\max_k P_{i,k}$.
5. For each expert $k$, surface the top features by $P_{i,k}$ with low $H(P_i)$ — these are the features expert $k$ owns.
6. Label the surfaced features with the normal SAE interpretability loop (top-activating examples + auto-interp).

## Why it matters

- **Finer than domain labels.** Where a domain-level probe might say "expert 3 handles code," SAE probing says "expert 3 handles *Python docstring openers*, *SQL SELECT clauses*, and *regex escape sequences*."
- **Diagnoses routing collapse.** If many high-$P_{i,k}$ features concentrate on a tiny subset of experts, expert balance is worse than router-entropy metrics would suggest. Useful signal for tuning [aux-loss-free-balancing](../architectures/aux-loss-free-balancing.md) and related balancing schemes.
- **Enables safety audits on MoE.** You can ask "is there an expert that specializes in refusal bypasses?" and get a direct answer instead of hand-labelling router outputs.

## Gotchas & tricks

- **Run at the *input* of the MoE layer, not the output.** The routing decision uses the input; probing the output confuses specialization with per-expert transforms.
- **SAE width matters.** Too few features and specializations alias into one unit; too many and dead features dominate. Start at $m / d = 16$ and sweep.
- **Features can be shared.** Many features fire across *several* experts — these are not "nobody's features," they are shared capabilities. Treat $H(P_i)$ as a continuous signal, not a binary.
- **Does not explain routing dynamics.** The method tells you *what* an expert correlates with, not *why* the router sends a given token there. For dynamic analysis you still need gradient-based attribution.
- **Expensive.** One SAE per MoE layer per checkpoint; ByteDance reports hours of GPU time per layer at 1B scale. Scaling to frontier MoE models is non-trivial.

## Sources

- Paper: *SAE Interprets MoE: Sparse AutoEncoders Reveal That MoE Experts Exhibit Human-Interpretable Fine-Grained Specialization* — Zhuo, Zhang, Lin, Lu — ByteDance Seed, 2026 — [arXiv:2611.08089](https://arxiv.org/abs/2611.08089).
- Background: *Scaling Monosemanticity* — Templeton et al., Anthropic, 2024 — the SAE recipe.
- Background: *DeepSeekMoE* — DeepSeek, 2024 — the fine-grained-expert MoE whose specialization patterns probes typically target. See [deepseek-moe](../architectures/deepseek-moe.md).
