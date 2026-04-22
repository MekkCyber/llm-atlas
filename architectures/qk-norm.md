# QK-Norm
*Depth — normalize queries and keys before the attention dot product.*

**TL;DR:** Apply RMSNorm to Q and K inside the attention block, after the projections but before `softmax(QKᵀ / √d)`. Bounds attention-logit magnitudes, which would otherwise drift upward at depth and at long training horizons, saturating softmax and collapsing gradients. Cheap, effective, now standard in stability-conscious LLMs.

**Prereqs:** [attention](../fundamentals/attention.md), [multi-head-attention](multi-head-attention.md), [transformer-block](transformer-block.md)
**Related:** [reordered-norm](reordered-norm.md), [z-loss](../fundamentals/z-loss.md), [_normalization](_normalization.md), [_training-stability](../pre-training/_training-stability.md), [olmo-2 case study](../case-studies/olmo-2.md)

---

## What it is

A tiny modification to the attention block: right after computing Q and K from the input and splitting them into heads, apply an RMSNorm **per head** to each before taking the dot product:

```
Q, K, V  = proj(x)       # standard projections
Q = reshape_heads(Q)     # [B, H, T, d_head]
K = reshape_heads(K)
Q = RMSNorm(Q)           # ← QK-norm, per head
K = RMSNorm(K)
(Q, K) = RoPE(Q, K)      # rotary position — AFTER the norm
A = softmax(Q @ K.T / √d_head)
out = A @ V
```

No change to the shapes, no learned mixing across heads. Two extra RMSNorm modules per block (one for Q, one for K), each with `d_head` learned scale parameters per head.

## How it works

The instability it fixes: during long training runs of deep models, the magnitudes of Q and K entries can drift upward (nothing explicitly penalizes them). When `‖Q‖ · ‖K‖` grows, the pre-softmax logits grow, and softmax becomes increasingly sharp — assigning probability ≈ 1 to a single token. The gradient of softmax at saturation is near-zero. The attention layer stops learning; downstream layers stop getting useful signal; loss spikes and the run dies.

QK-norm bounds `‖Q‖` and `‖K‖` to the learned RMSNorm scale. Since RMSNorm rescales to unit RMS (up to the learned scale γ), the product `‖Q‖ · ‖K‖` is held in a controlled range, keeping softmax at a usable temperature throughout training.

**Per-head is important.** Norming across heads (pre-split) would couple heads' magnitudes, partly defeating multi-head independence. Per-head RMSNorm preserves head specialization while still bounding each one.

**RoPE after QK-norm.** RoPE is a rotation, so it preserves norm. Applying RMSNorm then RoPE is equivalent in L2-norm to RoPE then RMSNorm, but the first ordering matches what every recent paper uses and avoids worrying about implementation details of complex-valued norms.

## Why it matters

- **Enables deeper models and longer training without spikes.** OLMo 2 credits QK-norm (with z-loss and reordered norm) as one of the three changes that closed the loss-spike failure modes from OLMo 1.
- **Almost free.** Two RMSNorms per block — a rounding error in FLOPs and parameters compared to attention and FFN.
- **No hyperparameter to tune.** Works out of the box.
- **Complements z-loss.** QK-norm bounds the attention-logit magnitude; z-loss bounds the output-logit magnitude. Different layers, same failure class.

## Gotchas & tricks

- **Place it in the right spot.** After Q/K projection and head reshape, before RoPE and the dot product. Putting it before the projection does nothing useful; putting it after the softmax is wrong.
- **Per-head RMSNorm, not per-token.** The norm is over the `d_head` feature dimension, per (batch, head, token). Sharing the scale across heads is a small quality hit.
- **γ initialization.** Start the RMSNorm scale at 1.0 — attention learns fine from that init. No need for fancy schemes.
- **Interaction with FlashAttention.** Most FlashAttention implementations accept pre-normed Q and K transparently. Just compute `q_norm = rmsnorm(q); k_norm = rmsnorm(k)` before calling the kernel.
- **Not a substitute for warmup.** Still use warmup. QK-norm prevents slow drift over thousands of steps, not the large gradients of the first few steps.

## Sources

- Paper: *Scaling Vision Transformers to 22 Billion Parameters* — Dehghani et al., 2023 — introduces QK-norm explicitly as a stability fix at depth.
- Paper: *Query-Key Normalization for Transformers* — Henry et al., 2020 — earlier exploration of the same idea.
- Paper: *2 OLMo 2 Furious* (OLMo 2 technical report) — AI2, 2024 — documents QK-norm as part of the stability package.
- Paper: *Gemma 2* — Google, 2024 — uses a variant of QK-norm.
