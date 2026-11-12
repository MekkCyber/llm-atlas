# Cross-Layer KV Sharing (DepthGraft)
*Depth — reuse shallow-layer K/V tensors at deep layers, trained end-to-end as structural regularization.*

**TL;DR:** Instead of each attention layer computing its own K and V, deep layers rebuild theirs from a **stride-aligned pair** of (shallow, middle) source layers via learnable per-channel coefficients. Gradient flow through the shared projections sharpens key retrieval directions at shallow layers and broadens value content; the resulting model trains to higher downstream accuracy *and* needs about one-third less KV cache at inference.

**Prereqs:** [multi-head-attention.md](./multi-head-attention.md), [../fundamentals/attention.md](../fundamentals/attention.md).
**Related:** [mla.md](./mla.md), [transformer-block.md](./transformer-block.md), [../inference/README.md](../inference/README.md).

---

## What it is

A training-time attention modification that treats cross-layer KV sharing as **regularization**, not just compression. The transformer stack is split into three stages — shallow, middle, deep. At each *deep* layer $\ell$:

- Pick a **shallow source** $\ell_s$ and a **middle source** $\ell_m$ by a fixed stride schedule.
- Compute a learned per-channel coefficient vector $\alpha_\ell$ (sigmoid of a scalar parameter per channel).
- Reconstruct: $K_\ell = \alpha_\ell \odot K_{\ell_s} + (1 - \alpha_\ell) \odot K_{\ell_m}$, likewise for $V$.
- The deep layer's query comes from its own $Q_\ell$ projection; only K and V are shared.

Nothing is tied across the shallow / middle layers — only the deep layers inherit.

## How it works

The paper's framing is that this is *regularization through gradient flow*. When deep layer $\ell$'s attention gradient flows back through the reconstruction, part of it lands on $(K_{\ell_s}, V_{\ell_s})$ and part on $(K_{\ell_m}, V_{\ell_m})$ — so shallow / middle K/V projections receive supervision from multiple downstream queries, not just their own.

The authors give two empirical signatures of this:

- **Shallow keys become more concentrated** in retrieval directions (effective rank drops).
- **Shallow values become more broadly distributed** across the content space (effective rank rises).

The stride-aligned pairing is deliberately sparse: spreading supervision thinly across many source layers avoids "anchor layer" collapse, where one shallow layer dominates every deep layer's K/V.

## Why it matters

- **Composable with existing KV-cache reductions.** DepthGraft works in both GQA and MLA stacks; the KV savings are *additive* because they come from a different axis (layer sharing vs head sharing).
- **Improves quality, not just memory.** Across 0.5B–16B dense and MoE models, DepthGraft adds downstream accuracy; +7.0 MMLU-Pro absolute on the 16B MoE. That's unusual for a KV-cache reduction — most of them pay a quality tax.
- **Inference throughput effects.** Prefill finishes early because deep layers can start as soon as their K/V sources are available. Decoding speed is nearly unchanged.
- **Reframes an inference trick as a training lever.** Suggests other inference shortcuts (e.g. layer skipping, token dropping) may also function as regularizers if trained end-to-end rather than bolted on.

## Gotchas & tricks

- **Stride schedule matters.** The paper's stride-aligned shallow+middle pairing beats single-source variants (deep ← shallow only) and all-to-one anchors. Deviating from the stride rule can erase the gains.
- **Not a drop-in for pretrained checkpoints.** The reconstruction coefficients are learned from scratch; naively applying the structure to a fully pretrained model without re-training degrades accuracy.
- **KV cache savings are ~1/3, not 1/2.** Only the deep layers reuse; shallow and middle layers still cache their own K/V.
- **Interacts with MLA's latent cache.** With MLA, each deep layer's shared K/V is already in latent space — DepthGraft's channel-wise coefficients operate there.

## Sources

- Paper: *DepthGraft: Structural Regularization through Hierarchical Cross-Layer KV Reconstruction* — Qin, Zhang, Wang, Wu, Xia, Yang, Yan, Shanghai Jiao Tong University / Dots Studio (Xiaohongshu), 2026 — [arXiv:2611.03142](https://arxiv.org/abs/2611.03142).
- Code: https://github.com/jianke0604/DepthGraft
