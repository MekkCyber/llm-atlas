# Softplus Attention Gate
*Depth — a smooth, unbounded gating function on attention outputs, replacing sigmoid or softmax-based gates.*

**TL;DR:** A per-head gate applied to attention outputs whose gating function is **softplus** — `g(x) = log(1 + exp(x))` — rather than a bounded sigmoid or a normalization-based gate. Because softplus is unbounded above, it avoids the "gate saturates at long context" failure mode of sigmoid gates while staying strictly positive and smooth. Introduced in K2-Horizon (MBZUAI, 2026) as one half of the attention-block changes that let a 36B-A4B MoVA model nearly match a 32B dense model at ~1/7 the active parameters.

**Prereqs:** [multi-head-attention.md](multi-head-attention.md), [transformer-block.md](transformer-block.md)
**Related:** [mova.md](mova.md), [qk-norm.md](qk-norm.md)

---

## What it is

A gating element inserted after the attention output (before the output projection) that reweighs each head's contribution per token:

```
attn_t = concat_h [ g(W_g^{(h)} · x_t) ⊙ attn_t^{(h)} ]
out_t  = W_O · attn_t
```

where `g(·)` is the softplus function. The gate is trained from scratch; there is no normalization or bounded squashing applied after `g`. The novelty is the functional form, not the position of the gate — gated attention blocks have been used since GLU and Gated Attention Unit; K2-Horizon's contribution is the softplus choice.

## How it works

Softplus `g(x) = log(1 + exp(x))` is a smooth, monotonically increasing, strictly positive function with:

- `g(x) → 0` as `x → −∞`
- `g(x) ≈ x` for `x ≫ 0`
- derivative `g'(x) = sigmoid(x)`, so gradient flow mirrors a standard sigmoid gate on the backward pass.

The key difference from `sigmoid(x)` as the gate is the *forward* shape: a sigmoid saturates at 1, so once a head is "fully on" the gate loses the ability to further amplify, which at long context tends to flatten the per-head distribution. Softplus has no upper bound, so heads whose logits grow at long context can keep amplifying rather than being capped.

Softmax-over-heads gating is a different alternative — it preserves scale but forces heads to compete. Softplus is non-competitive: heads can be simultaneously large.

Combined in K2-Horizon with [MoVA](mova.md): MoVA routes which value projection to use, the softplus gate reweighs the resulting per-head contributions. The pair sits in the attention block in place of the usual `W_O ∘ concat_heads`.

## Why it matters

- **No long-context saturation.** The reported motivation is that bounded gates (sigmoid) cap how aggressively a head can up-weight its contribution at long context, which the authors tie to attention-collapse symptoms at tens of thousands of tokens.
- **Keeps backward pass familiar.** The derivative is exactly sigmoid, so optimizer behavior and gradient magnitudes match a sigmoid gate — a drop-in replacement from an optimizer's perspective.
- **Cheap.** One extra linear per head plus a softplus, no normalization, no softmax across heads.

## Gotchas & tricks

- Softplus's unbounded output can let a single head dominate the block output. In K2-Horizon this is controlled by a residual-stream LayerNorm after the gated block, which normalizes the total; without an outer norm the scale drift can hurt training stability.
- At initialization, a bias on the pre-softplus linear chosen so `g` starts near a target gate value (e.g., 0.5 ≈ `log(1 + e^(−0.5))`) matters more than for a sigmoid gate because the function doesn't self-center.
- Don't combine with QK-norm and a bounded gate at the same time — the two interact to over-regularize head magnitudes. Pick one.
- The function choice is specific to K2; other smooth-unbounded choices (ELU+1, GELU with a positive shift) likely behave similarly but haven't been ablated in the paper.

## Sources

- Paper: *K2-Horizon: A Frontier Fleet of Open Source Language Models* — K2 team, MBZUAI, 2026 — [arXiv:2611.21003](https://arxiv.org/abs/2611.21003) — introduces the softplus attention gate.
- Related: GLU / Gated Linear Units family — the broader gating lineage this sits in.
