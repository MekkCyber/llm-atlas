# Virtual Head Attention (VHA)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A GQA-2 attention variant whose training-time expressiveness exceeds its inference cost. VHA halves the number of *physical* query heads and uses just 2 KV heads, but adds a **Q Premix** (linear expansion of queries into *virtual* heads) and a **Linear Postmix** (linear mix across heads). Because both operations are linear, they fold into the standard projection weights at inference — the deployed model is literally GQA-2 with no extra ops, yet beats regular GQA at every scale from 0.6B to 30B-A3B MoE.

**Prereqs:** [gqa.md](./gqa.md), [multi-head-attention.md](./multi-head-attention.md)
**Related:** [mla.md](./mla.md), [../fundamentals/attention.md](../fundamentals/attention.md)

---

## What it is

GQA caps attention expressiveness by the number of physical heads. VHA decouples expressiveness from head count by introducing *virtual* heads at training time:

- Physical heads: $H_Q^{\text{phys}}$ query heads, 2 KV heads.
- Virtual heads: $H_Q^{\text{virt}} > H_Q^{\text{phys}}$ via a linear expansion.

At inference, the virtual expansion collapses back into standard projection matrices and the deployed model is architecturally GQA-2.

---

## How it works

For input $X$:

1. **Q Premix.** Compute standard $Q = X W_Q$ of shape $[n, H_Q^{\text{phys}}, d_h]$. Linearly expand across the head axis with a small matrix $P \in \mathbb{R}^{H_Q^{\text{phys}} \times H_Q^{\text{virt}}}$:

$$
Q_{\text{virt}} = P^\top Q
$$

giving $H_Q^{\text{virt}}$ virtual query heads.

2. **Attention.** Run grouped-query attention with $Q_{\text{virt}}$ against the 2 KV heads, producing $H_Q^{\text{virt}}$ outputs.

3. **Linear Postmix.** Linearly mix the virtual-head outputs back to $H_Q^{\text{phys}}$ with a matrix $M \in \mathbb{R}^{H_Q^{\text{virt}} \times H_Q^{\text{phys}}}$.

4. **Standard output projection** $W_O$ as usual.

### Inference folding

Both $P$ and $M$ are linear operators acting on head axis. Compose them into the input $W_Q$ and output $W_O$ at inference:

$$
W_Q^{\text{inf}} = W_Q \cdot P, \quad W_O^{\text{inf}} = M \cdot W_O
$$

The deployed model has exactly the same operations as plain GQA-2. No extra FLOPs, no extra KV cache, no extra parameters stored.

---

## Why it matters

- **Pareto-dominates GQA at matched inference cost.** Reported across all tested scales (0.6B dense, 1.8B dense, 7B dense, 30B-A3B MoE) VHA beats GQA on standard benchmarks while reducing KV cache size and prefill/training FLOPs vs GQA baselines matched on parameter count.
- **Inference economics unchanged.** The virtual expansion is a training-time knob; the deployed model is still GQA-2. Serving infra needs no changes.
- **Design pattern is reusable.** "Add linear ops during training, fold them at inference" generalizes beyond heads — anything expressible as a factorized linear decomposition can be trained with the richer form and deployed with the compact form.

---

## Gotchas & tricks

- **Expansion ratio is a hyperparameter.** The paper's reported sweet spot is $H_Q^{\text{virt}} = 2 \cdot H_Q^{\text{phys}}$. Larger ratios recover MHA-level expressiveness but have diminishing returns.
- **Only linear premix/postmix fold.** Any nonlinearity between physical and virtual heads breaks the inference-folding trick — then VHA becomes a strictly bigger model at inference, defeating the point.
- **Compatible with RoPE and other positional encodings.** Rotary rotation applies to physical queries before Q Premix; the virtual expansion is a linear map on already-rotated queries.
- **Composes with MLA?** Open: VHA shrinks the head side; MLA shrinks the KV-latent side. In principle orthogonal, but no paper (as of 2026-10) explicitly combines them.

---

## Sources

- Paper: *Virtual Head Attention* — Ding, Wang, Zeng, Yang, Tao, Yang, Yu, Baidu / Peking University, 2026 — arXiv [2610.17959](https://arxiv.org/abs/2610.17959).
- Baseline: *GQA* — [gqa.md](./gqa.md).
- Design-space neighbor: *MLA* — [mla.md](./mla.md).
