# Looped Transformer
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A transformer whose *unrolled depth* exceeds its *stored depth*: the same shared block (or short stack of blocks) is applied for multiple rounds, letting the model do more sequential computation per token without storing more parameters. Attractive as a route to test-time reasoning via extra loop iterations. The catch is a subtle residual-scaling problem — because parameters are *tied* across visits, gradients aggregate and reads self-reference in a way that off-the-shelf DeepNorm doesn't correctly compensate for.

**Prereqs:** [transformer-block.md](./transformer-block.md), [_normalization.md](./_normalization.md)
**Related:** [reordered-norm.md](./reordered-norm.md) · [../pre-training/_training-stability.md](../pre-training/_training-stability.md)

---

## What it is

A conventional (untied) Transformer with physical depth $L$ has $L$ distinct parameter blocks and runs each block once. A looped Transformer keeps a smaller physical stack of $L_\text{phys}$ blocks and runs it for $R$ rounds, producing unrolled depth $N = L_\text{phys} \cdot R$. Compute scales with $N$; parameter count scales with $L_\text{phys}$.

The design lever is that reasoning-heavy tasks want more *sequential* computation per token, but not necessarily more parameters. Loops let you dial sequential depth at inference time (choose $R$) or trade depth for parameters at train time.

Two common instantiations:

- **Fully tied.** One block, applied $R$ times. Extreme parameter sharing.
- **Short-stack tied.** Small stack (2–4 blocks), applied $R$ times. Preserves some intra-cycle diversity while still sharing across cycles.

## How it works

A round is one pass through the physical stack. The residual stream accumulates across rounds:

$$
x_{r+1} = x_r + \text{Block}_{\theta}(x_r)
$$

with the *same* parameters $\theta$ read on every round. Two things change relative to an untied stack:

1. **Gradient aggregation.** During backprop, $\theta$ receives gradient contributions from every visit — the shared parameter aggregates $R$ terms per training step.
2. **Read self-reference.** The forward pass of round $r+1$ reads the same $\theta$ that round $r$'s gradient just updated (once optimizer step is applied) — a first-order feedback loop the untied case doesn't have.

DeepLoop (2026) formalizes this with a *visit-alignment coefficient* $\kappa_R$ that measures how correlated visits are in gradient direction. When visits are uncorrelated, standard DeepNorm ($\alpha \sim N^{1/4}$) still works. When visits are aligned — the conservative case that actually occurs during training — the residual-scaling exponent must double to $1/2$:

$$
\alpha = (2N)^{1/2}, \qquad \beta = (8N)^{-1/2}
$$

for unrolled depth $N$. This is the DeepLoop recipe: keep the Post-LN DeepNorm architecture but scale residuals for the *unrolled* depth, not the physical one.

The recipe is neutral when no physical block is revisited ($R = 1$: recovers standard DeepNorm) and improves validation loss and downstream accuracy once recurrent depth is activated.

## Why it matters

- **Sequential depth without parameter blowup.** A key candidate for test-time reasoning: crank $R$ at inference for hard prompts, drop it for easy ones.
- **Parameter-efficient depth-scaling.** For a fixed parameter budget, unrolled depth can be significantly deeper than a comparable untied stack.
- **Composes with mid-training.** A base can be trained with $R = 1$ and later fine-tuned into higher $R$, provided the residual scaling is set for the eventual $N$.
- **Recurrent depth is now stable.** Until DeepLoop, loop count was capped by training divergence rather than by evidence of diminishing returns.

## Gotchas & tricks

- **Residual scaling must target unrolled $N$, not physical $L_\text{phys}$.** Everyone's first mistake. If you plug DeepNorm's $\alpha$ with the physical count you'll appear to train fine and then diverge as you activate loops.
- **Gradient variance grows with $R$.** Even with correct scaling, per-parameter gradient variance rises as visits contribute more terms. Warmup and moderate LR are more important than in untied stacks.
- **KV cache at inference.** A looped block reads its own KV state across visits; caching strategies from untied stacks don't transfer trivially. Either recompute per visit or key the cache by (position, visit-index).
- **Position encodings interact with loops.** Absolute position IDs stay fixed across rounds, which is fine. Rotary encodings are also visit-invariant. But *learned* per-layer positional tables are meaningless in the looped setting — drop them.
- **Test-time $R$ is a real knob.** Reported gains on math / code come from letting $R$ grow at eval, not just at train. If evaluation reports don't say what $R$ was used, the number is under-specified.

## Sources

- Paper: *DeepLoop: Depth Scaling for Looped Transformers* — Li et al., Princeton / UC, 2026 — derives the visit-alignment coefficient and the 1/2 residual-scaling exponent.
- Paper: *Universal Transformers* — Dehghani et al., 2018 — the earliest tied-block Transformer; motivating precursor.
- Paper: *DeepNet: Scaling Transformers to 1000 Layers* — Wang et al., 2022 — the DeepNorm baseline whose exponent DeepLoop corrects.
