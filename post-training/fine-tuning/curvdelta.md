# CurvDelta: Low-Curvature Delta Modeling
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A fine-tuning / unlearning method that constrains weight updates to directions of **low curvature in the preservation loss**, so that updates barely move the model along the dimensions that preserve existing capabilities. Uses K-FAC to approximate the preservation-loss Hessian and a factorized projection to project updates into its low-curvature eigenspace. CurvLoRA is the low-rank variant. Better retain-vs-forget tradeoff than LoRA baselines on both math/code fine-tuning and knowledge unlearning (TOFU, MUSE).

**Prereqs:** [../grpo.md](../grpo.md) (background on loss landscape in LLMs), [../../fundamentals/attention.md](../../fundamentals/attention.md)
**Related:** [../../safety/README.md](../../safety/README.md)

---

## What it is

Fine-tuning collides with preserved capabilities when updates move through **high-curvature directions** of the preservation loss — small weight moves there produce large loss changes on the data you wanted to keep.

CurvDelta reframes the problem geometrically: instead of asking "which parameters to update," ask "which *directions* in parameter space are safe to update." Safe = low curvature of the preservation loss.

---

## How it works

Let $\mathcal{L}_P(\theta)$ be the loss on a preservation set (data representing capabilities you want to keep). Its Hessian $H = \nabla^2 \mathcal{L}_P$ has an eigenspectrum; directions with small eigenvalues are "flat" — moving there barely changes $\mathcal{L}_P$.

1. **Approximate $H$ with K-FAC** per layer: factored Kronecker approximation that scales to LLMs.
2. **Project the target-task gradient** $g_T = \nabla \mathcal{L}_T$ onto the low-curvature eigenspace of $H$:
   $$
   \tilde{g}_T = P_{\text{low-}\kappa} \cdot g_T
   $$
3. **Update using $\tilde{g}_T$** instead of $g_T$. The target task still gets optimized, but updates are confined to directions that minimally disturb $\mathcal{L}_P$.

**CurvLoRA**: same projection, applied inside a LoRA adapter's $A$, $B$ matrices — low-rank by construction.

For **unlearning**, swap $\mathcal{L}_T$ for a *forget* objective; the low-curvature constraint prevents the unlearning from spilling into retained knowledge.

---

## Why it matters

- **Unifies SFT, LoRA and unlearning** under one geometric principle. The same projection recipe applies regardless of whether you're adding or removing a capability.
- **Surgical unlearning.** On TOFU and MUSE, cleaner forget-vs-retain curves than Gradient Ascent, NPO and other unlearning baselines.
- **Better retention than plain LoRA.** On math and code fine-tuning, matches LoRA's target-task gain with less degradation of general ability.

---

## Gotchas & tricks

- **K-FAC approximation quality matters.** The whole method hinges on the Hessian estimate being accurate enough to separate flat from sharp. On small preservation sets or at low batch size the approximation can mis-rank directions.
- **Preservation set choice.** The method only preserves what $\mathcal{L}_P$ measures. For LLMs, this usually means a mix of general-language held-out data plus a sample of target downstream benchmarks. Choosing a narrow preservation set makes retention look better than it is.
- **Projection is per-layer.** The Kronecker factorization is layer-local; cross-layer curvature interactions are ignored. For most fine-tuning this is fine; for aggressive unlearning it can leak.
- **Doesn't eliminate catastrophic forgetting at scale.** Low curvature is relative; at high update magnitudes even flat directions move $\mathcal{L}_P$ enough to matter. Keep learning rates conservative.

---

## Sources

- Paper: *CurvDelta: Low-Curvature Delta Modeling For Surgical Learning and Unlearning* — Xiao, Liang, Chen, Wang, Chen, Lu — 2026 — [arXiv:2611.24654](https://arxiv.org/abs/2611.24654).
- Background: *K-FAC* — Martens & Grosse, 2015 — Kronecker-factored approximate curvature.
- Benchmarks: TOFU (Maini 2024) and MUSE (Shi 2024) — knowledge unlearning evaluation.
