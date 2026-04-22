# Training Stability

*Taxonomy — techniques that prevent loss spikes and training divergence at scale.*

**TL;DR:** Loss-spike failures at the 100B-token scale aren't random; they're symptoms of specific quantities drifting unboundedly during long runs (attention logits, output logits, residual-stream magnitude, gradient norms). Each stability technique bounds one of them. Modern stability-conscious training runs stack several — [QK-norm](../architectures/qk-norm.md) + [z-loss](../fundamentals/z-loss.md) + [reordered-norm](../architectures/reordered-norm.md) + gradient clipping + warmup — because the failure modes are largely independent. The OLMo 2 report is the most explicit public ablation of which technique closes which spike class.

**Related taxonomies:** [_normalization](../architectures/_normalization.md) · [_lr-schedules](_lr-schedules.md)
**Depth files covered here:** [qk-norm](../architectures/qk-norm.md) · [z-loss](../fundamentals/z-loss.md) · [reordered-norm](../architectures/reordered-norm.md)

---

## The problem

"Loss went to NaN at step 300k" is common lore in LLM pretraining, but it's not one failure — it's a family of them. Each one traces back to a specific quantity that training dynamics do not explicitly bound:

- Attention logits `QKᵀ` — softmax saturates, gradients collapse.
- Output logits — softmax saturates, numerical overflow in mixed precision.
- Residual-stream magnitude — sub-layer contributions become negligible at depth.
- Gradient norm per step — single huge update destroys the current weights.
- Adam's second-moment — goes to zero for unused parameters, large eps-related updates.

At small scale and short horizons these drift slowly enough to not matter. At scale and over weeks of training, any one of them can explode and end the run.

## The shared pattern

Every stability technique follows the same shape: **identify a quantity that could drift unboundedly, add a bound (soft or hard)**. The bound can be:

- **Architectural** — insert a normalization layer that physically can't let the quantity drift (QK-norm, RMSNorm, reordered-norm).
- **Loss-based** — add a regularizer term penalizing drift (z-loss).
- **Optimizer-side** — clip or rescale the update itself (gradient clipping, Adam epsilon).
- **Schedule-based** — pick a schedule that avoids the regime where drift is worst (warmup, LR decay).

Because the drifting quantities are largely independent, stability techniques are largely **orthogonal** — stacking them is additive, not redundant.

## Variants

| Technique | Bounds what | Where it lives | Stacks with others? |
| --- | --- | --- | --- |
| [QK-norm](../architectures/qk-norm.md) | Attention logit magnitude (`‖Q‖·‖K‖`) | Per-head RMSNorm on Q, K inside each attention block | Yes — orthogonal to output-side techniques |
| [Z-loss](../fundamentals/z-loss.md) | Output logit magnitude (`log Z`) | Auxiliary term `α·(log Z)²` added to CE loss | Yes — independent of attention-side techniques |
| [Reordered-norm](../architectures/reordered-norm.md) | Sub-layer contribution magnitude before residual merge | Norm placement inside the transformer block | Yes — compatible with QK-norm, z-loss |
| Pre-norm placement | Same as reordered but on sub-layer inputs | Norm placement inside the transformer block | Choose one of pre-norm or reordered |
| Gradient clipping (norm or value) | Per-step update magnitude | Between loss.backward() and optimizer.step() | Yes — universally applied |
| LR warmup | Large early updates when gradients are largest | LR schedule's opening phase | Yes — universally applied |
| Small-std init (< 0.02) | Initial activation magnitudes | Model initialization | Yes — foundational |
| Residual branch scaling (DeepNet, μP) | Residual-stream growth at depth | Per-layer scale on the residual branch | Alternative to pre-norm/reordered; combine carefully |
| Adam epsilon tuning | Numerical instability when second moment is tiny | Optimizer hyperparameter (default `1e-8`, sometimes `1e-15`) | Yes |
| Mixed-precision loss scaling | fp16 gradient underflow | Wraps loss.backward() | Yes — specific to fp16; bf16 doesn't need it |

## How to choose

**For a new training run at the 10B+ parameter, multi-trillion-token scale, stack the following by default:**

1. **[QK-norm](../architectures/qk-norm.md)** on Q and K inside attention.
2. **[Z-loss](../fundamentals/z-loss.md)** on final output logits (`α ≈ 1e-4`).
3. **Pre-norm or [reordered-norm](../architectures/reordered-norm.md)** placement (pick one).
4. **RMSNorm** as the norm layer.
5. **Gradient clipping** at norm = 1.0 or similar.
6. **LR warmup** over the first 0.5–2% of tokens.
7. **Small-std init** (0.02 or tighter, residual branch scaled).
8. **bf16** mixed precision (no loss-scaling needed) or fp8 with its own instability-management (separate topic).

This is roughly the [OLMo 2](../case-studies/olmo-2.md) recipe, which was ablated to show each piece closes a distinct spike class.

**Under tighter scale (1B params, 100B tokens), most techniques are unnecessary** — spikes are rare because drift doesn't accumulate long enough. QK-norm and z-loss are still cheap insurance.

**When to add more:** if a run spikes at step K, diagnose which quantity drifted (attention logit magnitude? output logit? residual stream?) before adding ad-hoc fixes. The point of a structured taxonomy is that the failure mode maps to a specific technique.

### What not to do

- **Don't skip warmup.** It's the single cheapest stability trick and often the only one that matters at small scale.
- **Don't combine multiple residual-scaling schemes blindly** (e.g., both DeepNet scaling and reordered-norm). They can over-suppress sub-layer contributions and kill effective depth.
- **Don't treat loss spikes as "restart and hope"** once they show up. Restarting from a pre-spike checkpoint with the same recipe gives the same spike. Diagnose, add the missing bound, then restart.

## Adjacent but distinct

- **[Normalization](../architectures/_normalization.md) taxonomy.** Where to put the norm and which norm to use. Overlaps with this page via QK-norm and reordered-norm but focuses on the norm layer itself.
- **[LR schedules](_lr-schedules.md).** Warmup and LR-decay choices intersect with stability but the schedule is a broader design space.
- **Optimizer choice** (AdamW vs. Lion vs. Muon). Influences stability but is usually discussed as a separate axis.
- **Mixed precision** (fp16 / bf16 / fp8). Precision-specific stability (loss scaling, FP8 calibration) is its own topic.

## Sources

- Paper: *2 OLMo 2 Furious* — AI2, 2024 — most explicit public ablation of QK-norm + z-loss + reordered-norm as a stability package.
- Paper: *PaLM: Scaling Language Modeling with Pathways* — Chowdhery et al., 2022 — introduces z-loss for LLM stability.
- Paper: *Scaling Vision Transformers to 22 Billion Parameters* — Dehghani et al., 2023 — QK-norm for depth stability.
- Paper: *DeepNet: Scaling Transformers to 1,000 Layers* — Wang et al., 2022 — residual branch scaling approach (post-norm-compatible).
- Paper: *On Layer Normalization in the Transformer Architecture* — Xiong et al., 2020 — theoretical analysis of placement choices.
- Paper: *The Llama 3 Herd of Models* — Meta, 2024 — production-scale stability recipe notes.
