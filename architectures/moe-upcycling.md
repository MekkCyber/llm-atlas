# MoE Upcycling (with Rotation-Noise Symmetry Breaking)

*Depth — how to expand a smaller MoE (or dense) model into a bigger MoE by duplicating experts, and why naive duplication fails without rotation noise.*

**TL;DR:** Training a new large MoE from scratch is expensive. **Upcycling** starts from an already-trained model and expands it in depth (repeating transformer blocks) and width (duplicating experts and their router rows), then continues training on a large token budget. The subtle bit: raw duplication leaves each pair of experts *exactly tied* — same weights, same routing bias, so gradients are identical, they never diverge, and the added capacity is wasted. K-EXAONE 2.0 breaks symmetry with **norm-preserving random rotations** on duplicated experts, letting the two copies drift into different specializations while preserving the parent model's activation statistics.

**Prereqs:** [_moe.md](_moe.md), [deepseek-moe.md](deepseek-moe.md)
**Related:** [aux-loss-free-balancing.md](aux-loss-free-balancing.md) · [capacity-factor.md](capacity-factor.md) · [../pre-training/mid-training.md](../pre-training/mid-training.md) · [../case-studies/k-exaone-2.md](../case-studies/k-exaone-2.md)

---

## What it is

Upcycling = start from a trained parent and initialize a larger child by structured duplication:

- **Depth expansion.** Insert repeated copies of existing transformer blocks. K-EXAONE 2.0's block unit is `LLLG` (three local sliding-window layers + one global) and they expand 12 → 19 blocks (48 → 78 layers) by repeating middle-of-stack blocks.
- **Width expansion (experts).** Duplicate each expert and its router-row so `E_child = 2 · E_parent` (128 → 256 in K-EXAONE 2.0). Router biases are preserved; the inherited bias update keeps load balanced across the new expert set.
- **Tokenizer + attention shapes:** inherited unchanged.

Continued training on a large token budget (8T tokens in the K-EXAONE 2.0 case) then specializes the new capacity.

## How it works — the symmetry-breaking trick

Duplicating an expert `E → (E, E)` with identical router rows is a *degenerate* initialization:

- Both copies compute the same output on any input.
- They receive the same routing probability, so they see the same tokens.
- Their gradients are identical, so they update in lockstep.
- Without symmetry breaking, the two copies remain exactly tied forever — the "expansion" adds parameters but no effective capacity.

**Fix (K-EXAONE 2.0):** add a **norm-preserving random rotation** to one of the duplicated experts:

- Sample a random orthogonal matrix `R` per duplicated expert.
- Apply `R` to the expert's projection weights so the mapping is `x → E(x) → R · E(x)` (with a compensating inverse rotation folded into downstream weights, or the rotation is applied to internal FFN latents where it doesn't change the output distribution's norm).
- The two copies now compute different outputs on the same input, receive different gradients, and can specialize.

"Norm-preserving" matters: unlike additive Gaussian noise on weights, an orthogonal rotation does not change the magnitude of the activation, so the parent model's *activation statistics* are inherited exactly at step 0. Training doesn't need a stabilization warm-up to recover the parent's numerical regime.

## Why it matters

- **Cheaper than training from scratch.** Reuses the parent's already-learned features; only continued training is needed to specialize new experts. K-EXAONE 2.0 gets 3× the active-parameter capacity of its parent for a fraction of a from-scratch training budget.
- **Preserves training dynamics.** Norm-preserving initialization means loss and gradient norms at step 0 look like the parent's — no destabilization from added parameters.
- **Compatible with aux-loss-free balancing.** Because router biases are inherited unchanged and duplicated symmetrically, the standard balancing bias update maintains load across the expanded expert set with no intervention.

## Gotchas & tricks

- **Naive weight-noise doesn't preserve norm.** Additive Gaussian noise on FFN weights breaks symmetry but changes activation magnitude — either lengthens warmup or destabilizes training. Random orthogonal rotation is norm-preserving by construction.
- **Depth expansion must respect residual structure.** Repeating blocks from the middle of the stack (K-EXAONE 2.0's choice) preserves the parent's residual-stream statistics better than duplicating near the boundaries.
- **You still need the compute.** Upcycling saves on the initial "learn language" cost but the 8T-token continued training in K-EXAONE 2.0 is not free — plan for a serious continued-pretraining budget.
- **Doesn't fix architectural mismatches.** If the child needs a *different* number of attention heads or head-dim, you're back to from-scratch. Upcycling is expansion, not remodeling.
- **Router capacity assumptions carry over.** If the parent was undertrained on load balancing, doubled experts don't fix that — check inherited router health before expanding.

## Sources

- Paper: *K-EXAONE 2.0 Technical Report* — LG AI Research, 2026 — [arXiv 2608.04505](https://arxiv.org/abs/2608.04505). Introduces the rotation-noise trick and the LLLG-block depth-expansion recipe.
- Earlier: *Sparse Upcycling: Training Mixture-of-Experts from Dense Checkpoints* — Komatsuzaki et al., 2022. Foundational dense→MoE upcycling; K-EXAONE 2.0 extends the technique to MoE→MoE with rotation noise.
