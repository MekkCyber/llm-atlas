# U-Space
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A low-dimensional **subspace of activations** that tracks how an LLM's uncertainty evolves during generation. A readout — the **U-Lens** — produces token-level uncertainty maps and a scalar score, with no correctness labels, no repeated sampling, and no training. Beats established uncertainty baselines on reasoning benchmarks under standard *and* length-controlled evaluation, and transfers across models more reliably than supervised uncertainty estimators.

**Prereqs:** [README.md](./README.md), [../fundamentals/attention.md](../fundamentals/attention.md)
**Related:** [../post-training/reasoning/README.md](../post-training/reasoning/README.md)

---

## What it is

An identified low-dimensional subspace, inside the hidden activations of a decoder LLM, whose coordinates linearly track a calibrated notion of uncertainty. "U-Lens" is the projection-plus-readout that turns an activation into (a) a token-level uncertainty heatmap along the generation and (b) a single scalar uncertainty score for the sequence.

## How it works

- **Identify the subspace.** From a modest set of generations, compute the activation statistics that correlate with model confidence and extract a low-rank subspace. The paper reports this requires neither correctness labels nor repeated sampling.
- **Project at inference.** For each generated token, project its hidden state into U-Space and read off the uncertainty coordinates.
- **Aggregate.** Collapse per-token readings into a scalar sequence score (mean, max, or paper-specific weighting).

Because the subspace is identified from the model's own geometry rather than fit to a labelled task, the readout transfers across tasks and across models in the same family.

## Why it matters

Standard uncertainty estimators either (a) sample N times and look at consistency, which multiplies inference cost, or (b) train a supervised head, which needs labels and overfits. U-Space is a *free* signal: one forward pass, no labels, no retraining. The reasoning-benchmark result that it survives length-controlled evaluation matters because many "confidence drops with length" findings are confounded by sequences simply being longer; U-Space's advantage holds net of that.

## Gotchas & tricks

- Projection matrices are model-specific; U-Lens for Llama-3 is not U-Lens for Qwen3 even if the recipe for finding them is the same.
- Token-level maps are easy to over-interpret. The uncertainty signal reflects the model's state, not a notion of ground-truth difficulty.
- The method is activation-based, so quantised or distilled derivatives of a base model need re-identification.
- Combines naturally with selective decoding (abstain, backtrack, request human review) and with reasoning-step gating (replan when uncertainty spikes).

## Sources

- Paper: *U-Space: Uncovering When and Why Uncertainty Arises in Language Models* — Loose et al., TU Darmstadt / UCL / Mila / MBZUAI, 2026 — [arXiv:2610.09087](https://arxiv.org/abs/2610.09087)
