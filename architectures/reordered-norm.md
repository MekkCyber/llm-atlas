# Reordered Norm
*Depth — OLMo 2's norm placement: normalize sub-block outputs, leave the residual stream clean.*

**TL;DR:** A variant of the transformer block where RMSNorm is applied to the **output** of attention and FFN *before* they're added to the residual stream, rather than to their inputs (pre-norm) or to the post-residual sum (post-norm). The residual stream carries unnormalized signal end-to-end, while each sub-block's contribution is bounded. Empirically the most stable of the three placements at scale.

**Prereqs:** [transformer-block](transformer-block.md)
**Related:** [qk-norm](qk-norm.md), [z-loss](../fundamentals/z-loss.md), [_normalization](_normalization.md), [_training-stability](../pre-training/_training-stability.md), [olmo-2 case study](../case-studies/olmo-2.md)

---

## What it is

Three ways to place norm around a sub-layer:

```
# Post-norm (original AIAYN, 2017)
x_new = Norm(x + Sublayer(x))

# Pre-norm (GPT-2 onward, almost universal modern default)
x_new = x + Sublayer(Norm(x))

# Reordered norm (OLMo 2)
x_new = x + Norm(Sublayer(x))
```

The key difference: in reordered-norm the **residual stream `x` is never touched by a norm**, and the **sub-layer is computed on un-normed input** but its output is normed before being added back.

Applied to both sub-layers of a standard block:

```
x₁ = x  + Norm(Attention(x))
x₂ = x₁ + Norm(FFN(x₁))
```

## How it works

The instability each placement faces:

| Placement | What grows unbounded | Failure mode |
|---|---|---|
| Post-norm | Residual and gradient scale grow before the norm | Hard to train at depth without careful warmup |
| Pre-norm | Residual stream magnitude grows across layers (no norm on the skip path) | Late layers see huge residual, attention/FFN contributions become negligible, effective depth collapses |
| Reordered | Sub-layer inputs can drift, but outputs are bounded before merging | Most balanced — residual stays meaningful, sub-layer contributions stay bounded |

Reordered-norm inherits pre-norm's **clean residual gradient** (no norm on the backward path through the skip connection) because the residual `x` passes through the identity. It also gets a property pre-norm lacks: **each sub-layer contribution is bounded before addition**, so the residual can't run away from accumulated sub-layer drift.

### Why not just use pre-norm?

Pre-norm's residual-growth problem is subtle but real at scale. As layers stack, the un-normed residual accumulates magnitude, the next `Norm(x)` rescales it down for the sub-layer but the sub-layer's output goes back to the large residual. The ratio of sub-layer contribution to residual shrinks with depth — later layers effectively act like identity maps. Reordered-norm keeps the sub-layer's contribution on a controlled scale, so later layers still contribute.

### Why not just use post-norm?

Post-norm normalizes the whole residual + sub-layer sum, which is equivalent to putting a norm on the residual path. That makes backprop through many layers pass through many norms, producing vanishing or exploding gradients at depth. It's known to require aggressive warmup and is fragile beyond ~20 layers without tricks. Reordered-norm avoids this entirely.

## Why it matters

- **Lets you train deeper / longer without spikes.** Combined with [qk-norm](qk-norm.md) and [z-loss](../fundamentals/z-loss.md), closes the common "loss suddenly diverges at step 50k" failure mode.
- **Cheap change.** Just moves the norm — same parameters, same FLOPs, same module count.
- **Stability contribution is independent.** In the OLMo 2 ablations, swapping pre-norm → reordered-norm alone removes a distinct class of slow-drift spike that QK-norm and z-loss don't fully cover.

## Gotchas & tricks

- **Don't confuse with post-norm.** Post-norm wraps the residual sum; reordered-norm wraps only the sub-layer output. Easy to miscode.
- **Final layer norm still needed.** Apply one RMSNorm to the residual stream before the LM head, same as pre-norm models do.
- **Initialization.** RMSNorm γ = 1.0 works. No special init of residual branch scaling required (in contrast with post-norm where DeepNet-style scaling is sometimes needed).
- **Not the same as "sandwich norm".** Sandwich norm (Cogview) applies *two* norms — one on input and one on output of each sub-layer. Reordered-norm is one norm on the output only.
- **Interaction with residual scaling.** If you additionally scale residual branches (e.g., dividing by √depth), measure the effect — the combination can over-suppress sub-layer contribution.

## Sources

- Paper: *2 OLMo 2 Furious* — AI2, 2024 — introduces the reordered placement and ablates it against pre-norm.
- Paper: *On Layer Normalization in the Transformer Architecture* — Xiong et al., 2020 — theoretical analysis of pre- vs. post-norm.
- Paper: *CogView* — Ding et al., 2021 — sandwich-norm variant for comparison.
