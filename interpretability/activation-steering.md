# Activation Steering
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** At inference time, **add a vector to a model's residual stream** at a chosen layer to shift its generation toward or away from a concept — "steering" the model without any weight update. The vector is derived from contrasts between activations on concept-positive and concept-negative inputs. Works for text generation, and (Zeng et al. 2026) also for image generation inside unified multimodal models — see [cross-modal-steering.md](cross-modal-steering.md).

**Prereqs:** *(none)*
**Related:** [cross-modal-steering.md](cross-modal-steering.md)

---

## What it is

A **steering vector** $v_c$ for concept $c$ is a direction in the residual stream such that adding $\alpha \cdot v_c$ at inference time (for some scalar $\alpha$) biases generation toward (or, with negative $\alpha$, away from) that concept. The technique sits between prompting (zero cost, soft) and fine-tuning (persistent, expensive).

Core observation: concept representations in modern transformers are often *linearly separable* at some intermediate layer. A single vector added at that layer can shift behavior reliably across prompts.

## How it works

### Deriving the vector

Common recipes:

- **Mean difference.** Collect activations $h_i^+$ (positive examples: outputs instantiating the concept) and $h_i^-$ (negative examples). The steering vector is $v_c = \mathrm{mean}(h^+) - \mathrm{mean}(h^-)$. Simplest, surprisingly effective.
- **Difference of two prompt completions.** For a single contrastive pair ("tell me a happy story" vs "tell me a sad story"), the layer-$\ell$ residual-stream difference *is* the steering vector. See *activation addition* (Turner et al.).
- **SAE direction.** Train a sparse autoencoder on the model's activations; identify a feature direction corresponding to the concept, and use that as $v_c$.
- **Probing direction.** Train a linear probe for the concept; its weight vector is $v_c$.

### Applying the vector

At inference, pick a layer $\ell$ (typically mid-depth). Compute the residual stream $h_\ell$ as usual, then **replace** it with $h_\ell + \alpha \cdot v_c$ before passing to layer $\ell + 1$. $\alpha$ in the range of $0.5$ to $5\times$ the natural norm of $v_c$ is common.

Variants:

- **All-token vs last-token-only.** Steer every position vs. only the generation positions.
- **Per-layer vs single-layer.** Add at one layer vs. a schedule across several.
- **Direction clamping.** Project $h_\ell$ onto $v_c$ and clamp its magnitude.

## Why it matters

- **No training.** Works from a frozen model with a handful of contrastive examples. Deployable instantly.
- **Reversible and composable.** Multiple steering vectors can be added simultaneously (within reason). Removing one is one subtraction.
- **A probe into representation structure.** The fact that single directions reliably control behavior is evidence of linear concept encoding — a core finding of modern mech-interp.
- **Shared across modalities inside unified models.** Steering vectors derived in one modality can steer another — see [cross-modal-steering.md](cross-modal-steering.md).

## Gotchas & tricks

- **Layer choice matters.** Too early: not enough concept representation. Too late: not enough downstream residual depth to act. Sweep layers on a small eval; the best layer is usually mid-depth.
- **Magnitude is a lever.** Low $\alpha$ barely steers; high $\alpha$ breaks fluency. Expect an optimum.
- **Context collapse.** Strong steering can override everything about the prompt (asking "what time is it?" with a strong "happy" vector may generate joy-themed irrelevance). Not just a bug; it's informative about where the vector pulls.
- **Not fine-tuning.** Steering is on top of a frozen model; its ceiling is what the model can already do. For capabilities the model doesn't have, no vector exists.
- **Depends on linear concept geometry.** Some concepts (e.g. multi-faceted ones) don't live in a single direction. SAE features sometimes carve the space better than mean-difference.
- **Safety-adjacent.** Steering is a dual-use mech-interp tool: it can suppress harmful generations and also *induce* them. Access controls matter for deployed steering endpoints.
- **Attribution note.** The idea precedes this entry — see "activation addition" (Turner et al. 2023), representation engineering (Zou et al. 2023), and the SAE-based steering line (Templeton et al. 2024). This page covers the technique as a class; [cross-modal-steering](cross-modal-steering.md) is the specific transfer result that triggered its creation.

## Sources

- Paper: *Activation Addition: Steering Language Models Without Optimization* — Turner, Thiergart, Udell, et al., 2023 — the mean-difference recipe.
- Paper: *Representation Engineering: A Top-Down Approach to AI Transparency* — Zou et al., 2023 — broader framework including steering.
- Paper: *Scaling Monosemanticity* — Templeton et al., Anthropic, 2024 — SAE-feature-direction steering at frontier scale.
- Paper: *Cross-modal Steering Vectors Exist Across Both Image and Text Generation* — Zeng, Subramani, Rott Shaham, Krojer, et al., 2026 — [arXiv:2610.36269](https://arxiv.org/abs/2610.36269)
