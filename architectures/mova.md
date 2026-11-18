# Mixture-of-Value Attention (MoVA)
*Depth — a mixture-of-experts applied inside the attention block, routing on the value projection.*

**TL;DR:** MoVA brings MoE into the attention block itself, specifically to the **value projection**. Instead of one `W_V` per head, each attention head owns a small *pool* of value experts, and a token-dependent gate picks top-*k* of them per head per token. The output for that head becomes the gated sum of the selected experts' values, used in the standard `softmax(QK^T/√d) · V` formula. Introduced in K2-Horizon (MBZUAI, 2026) alongside a softplus attention gate; the 36B-A4B configuration using both nearly matches a dense 32B trained on the same data with **one-seventh** the active parameters.

**Prereqs:** [multi-head-attention.md](multi-head-attention.md), [_moe.md](_moe.md)
**Related:** [softplus-attention-gate.md](softplus-attention-gate.md), [mla.md](mla.md), [deepseek-moe.md](deepseek-moe.md)

---

## What it is

A mixture-of-experts over **attention value projections**. In vanilla multi-head attention each head owns a single `W_V ∈ R^{d_model × d_h}`; MoVA replaces this with `E` candidate value projections per head, `{W_V^{(h,e)}}_{e=1..E}`, plus a lightweight router `g_h(x_t)` that selects top-*k* of them for each token. Queries and keys are unchanged; the full budget of experts lives only in V.

This is architecturally distinct from an FFN-side MoE: the experts are *inside attention*, operate per-head, and the routing happens *before* softmax, so each token's attention weights are computed against standard K and then applied to a token-specific value basis.

## How it works

For head *h* at position *t*:

```
scores_{h,t,e}   = g_h(x_t)[e]                           # router output
S_{h,t}          = top-k(scores_{h,t,·})                 # chosen expert indices
w_{h,t,e}        = softmax(scores_{h,t,·})[e]  for e ∈ S_{h,t}
V_t^{(h)}        = Σ_{e ∈ S_{h,t}} w_{h,t,e} · W_V^{(h,e)} · x_t

attn_t^{(h)}     = softmax(Q^{(h)} K^{(h)⊤} / √d_h) · V^{(h)}
```

Only V is expert-routed; Q and K use standard per-head projections. The gating uses the same balancing machinery as FFN MoE (auxiliary-loss-free bias, or aux loss) to keep experts loaded evenly. Active parameter count is `d_model × d_h × k` per head, while the total parameter count is `d_model × d_h × E` per head — so the compute cost rises only with top-*k*.

The intuition: attention weights decide *which positions to attend to*, but the value projection decides *what information to carry forward*. Giving the latter per-token specialization is a sparser use of parameters than giving every head its own giant `W_V`.

## Why it matters

- **Decouples attention routing from value routing.** Attention weights still reflect position-level relationships; value routing adds a token-type-specialization axis orthogonal to that.
- **Fits into standard attention kernels.** The only change is a gather of `V` from the selected experts; Q/K/softmax code is untouched.
- **Scales active vs total parameters independently.** Lets a 36B-A4B configuration using MoVA + softplus attention gate nearly match a dense 32B at **~1/7 the active parameters** per K2-Horizon.

## Gotchas & tricks

- Balancing matters more than in FFN MoE: token-type collapse manifests as a few experts always firing per head, which degenerates MoVA back to multi-head attention. Pair with aux-loss-free balancing or a per-head sequence-wise balance loss.
- Memory layout of `{W_V^{(h,e)}}` is tricky; the natural layout collides with standard attention kernels. The K2 implementation reportedly groups experts per head, not per layer, to keep gathers tight.
- Routing on V only is a design choice — K2 discusses that routing on K or Q leaked position information through the gating signal, which hurt small-model quality.
- Don't confuse MoVA with "MoE-Attention" papers that route the *heads* themselves. MoVA routes among value projections *within* a head; head count is fixed.

## Sources

- Paper: *K2-Horizon: A Frontier Fleet of Open Source Language Models* — K2 team, MBZUAI, 2026 — [arXiv:2611.21003](https://arxiv.org/abs/2611.21003) — introduces MoVA.
- Context: [DeepSeek-MoE](deepseek-moe.md) for the fine-grained-expert MoE pattern MoVA borrows balancing from.
