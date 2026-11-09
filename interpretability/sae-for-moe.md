# SAEs for Mixture-of-Experts
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Train a sparse autoencoder on the output of a Mixture-of-Experts layer, then attribute each SAE feature to the experts that activate it. The result is a fine-grained map of *which expert produces which concept* — which turns out to be far sharper than what router-weight analysis or domain-label clustering had previously suggested. Introduced in 2611.08089 on Qwen3-30B-A3B.

**Prereqs:** [sparse-autoencoder](sparse-autoencoder.md), [_moe](../architectures/_moe.md)
**Related:** [deepseek-moe](../architectures/deepseek-moe.md)

---

## What it is

A two-step pipeline that composes SAE feature discovery with MoE expert attribution:
1. Train an SAE on a MoE layer's output activation $h^{\text{out}}_{\text{MoE}}$ (after summing contributions of the selected experts).
2. For each SAE feature, measure which experts' contributions dominate when that feature is active — an **expert-to-feature ownership map**.

Applied in the source paper to layers 4, 30, and 46 of Qwen3-30B-A3B (early, mid, late depth).

## How it works

Setup: MoE layer with $N$ experts, top-$K$ routing. For token $t$, the output is $h^{\text{out}}_t = \sum_{i \in \text{Top-}K(t)} g_{i,t} \cdot \text{FFN}_i(h_t)$.

1. **Train SAE** on $\{h^{\text{out}}_t\}$ with the usual sparsity + reconstruction objective. Obtain feature directions $\{d_f\}$.

2. **Per-expert attribution.** For each feature $f$ and each expert $i$, compute the average contribution of expert $i$ to feature $f$'s activation:
$$A_{f, i} = \mathbb{E}_t\big[\mathbb{1}[z_{f,t} > 0] \cdot \langle g_{i,t} \cdot \text{FFN}_i(h_t), d_f \rangle\big]$$

3. **Measure concentration.** For each feature, rank experts by $A_{f, i}$ and report the top-1 share, top-$k$ share, and entropy.

Result on Qwen3-30B-A3B: many features show near-one-to-one feature-to-expert ownership — the top expert accounts for most of the feature's activation, with the pattern holding across layers 4, 30, 46.

## Why it matters

- **Validates fine-grained MoE design.** Papers like DeepSeekMoE argued that many small experts yield stronger specialization than few big ones; SAE attribution confirms this empirically at the feature level.
- **Interpretable targets for intervention.** Expert-level pruning, steering, or safety analysis get a semantic meaning — turn off "expert 23 at layer 30" ≈ turn off a known concept.
- **Bridges SAE and MoE literatures.** Previously SAEs were analyzed on dense layers; this extends the methodology to sparse layers.

## Gotchas & tricks

- Attribution is measured on *active* features only; dead SAE atoms don't give useful signal.
- **Routing stochasticity.** Top-$K$ routing is argmax, so attribution is well-defined, but noisy-top-$K$ (common in some MoE variants) adds variance — average over many tokens.
- **Layer choice matters.** Early layers (4) tend to have token-surface features; late layers (46) have abstract concepts; mid-layers (30) are the richest for ownership-style analysis.
- Doesn't say the inverse: that each expert owns a *single* feature. Experts own clusters of related features, and features can co-fire across experts in some cases.

## Sources

- Paper: SAE Interprets MoE: Sparse AutoEncoders Reveal That MoE Experts Exhibit Human-Interpretable Fine-Grained Specialization — Zhuo et al., 2026 — [arXiv:2611.08089](https://arxiv.org/abs/2611.08089)
