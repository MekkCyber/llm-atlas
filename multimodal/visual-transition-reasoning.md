# Visual Transition Reasoning (WOVEN)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Argues that spatial, embodied, physical, and temporal reasoning in multimodal LLMs all bottleneck on the same underlying capability: predicting the *visual transition* from (image, action) to (image). WOVEN is a 36,076-example training-and-benchmark set generated from video rollouts, organized by scene × action × reasoning type. Training on WOVEN subsets improves 22 of 26 external benchmarks, and a 38-model evaluation shows large gaps from human performance.

**Prereqs:** [multimodal/README](../multimodal/README.md)
**Related:** [_data-curation](../data/_data-curation.md)

---

## What it is

A reusable training signal and benchmark for multimodal LLMs built around before/after pairs with controlled action semantics. Rather than one new capability, it treats visual-transition prediction as a *primitive* that unlocks a family of downstream reasoning tasks.

## How it works

1. **Video rollout generation.** Use video models / simulators to generate rollouts with controlled scenes and actions.
2. **Transition extraction.** Pull before/after pairs plus action descriptions, keeping (image_before, action, image_after) triples.
3. **Categorization.** Each example gets tagged by (scene type, action type, reasoning type: spatial / physical / embodied / temporal).
4. **Benchmark + training split.** Evaluation uses held-out scene–action combinations; training uses the rest.
5. **Supervision modes.** Multiple objectives tested — predict image_after given (image_before, action), predict action given both, VQA-style questions about the transition.

The organizational key is the scene × action × reasoning-type grid, which lets the paper derive a recipe for which subsets to train on for maximum downstream transfer.

## Why it matters

- **Unifies four MLLM weaknesses.** Spatial, embodied, physical, and temporal reasoning share a core deficit in visual-transition prediction; targeting it hits all four at once.
- **Large held-out gap vs humans.** Across 38 frontier models, every one shows substantial room for improvement — this is a *live* capability axis, not a saturated one.
- **Measurable transfer.** Training on WOVEN subsets improves 22 of 26 *external* benchmarks across the four reasoning categories — strong evidence the signal is reusable, not just in-distribution.

The paper also derives a selection recipe (which (scene, action, reasoning-type) cells to train on for which downstream capability), making this directly actionable for MLLM post-training.

## Gotchas & tricks

- Depends on controllable video generation — the quality of the before/after pairs ceilings the signal.
- The 22-of-26 transfer number is on the paper's eval set composition; coverage of categories matters when generalizing the claim.
- Pure next-frame prediction underperforms the mixed objective suite — the auxiliary VQA signal is doing meaningful work.
- Temporal reasoning benefits most, physical reasoning least, in the paper's ablations.

## Sources

- Paper: *WOVEN: Weaving Visual World Modeling into Multimodal LLMs* — Fan et al., Northwestern/CMU/UNC/Amazon, 2026 — [arXiv:2610.12417](https://arxiv.org/abs/2610.12417)
