# Confident Layer Decoding
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Aligned LLMs exhibit a *Guess–Refine–Perturb* pattern across layers — early layers guess, middle layers refine, **final layers can perturb toward alignment-preferred tokens at the cost of reasoning quality**. Confident Decoding (2026) is a training-free decoding rule that walks backward from the final layer and accepts the first layer whose logit distribution is *confidently refined* by an entropy criterion. Consistent gains on GPQA-Diamond, Omni-MATH, and HLE with <2% latency overhead and zero memory cost.

**Prereqs:** [README.md](README.md)
**Related:** [../interpretability/README.md](../interpretability/README.md), [../safety/refusal-suppression.md](../safety/refusal-suppression.md), [../post-training/_post-training.md](../post-training/_post-training.md)

---

## What it is

Standard autoregressive generation reads off the *final* layer's logits, assuming deeper is better. The paper shows that for aligned (RLHF / DPO-tuned) models, this is wrong in a predictable way: alignment fine-tuning concentrates in the last block or two, where the model "rounds" coarse-but-correct reasoning predictions toward refusal-flavoured or generic-helpful tokens.

Confident Decoding bypasses those final layers at inference time, picking a near-final layer whose distribution still reflects the reasoning the model actually did.

## How it works

Frame layer selection as an *optimal stopping problem*: walk backward from layer `L` (final) to layer `L − k`. At each candidate layer, project the residual stream through the unembedding and measure the entropy of the resulting logit distribution.

- If entropy is *low* (the layer is confident) and the top-1 token would change vs. the final layer's pick, accept this layer's logits.
- If entropy is *high* (not confident), keep walking back; eventually fall through to the final layer.

Under bounded projection noise and dominant late-stage alignment perturbation, the rule provably bounds the loss relative to an oracle that knows the best refinement layer.

Compute cost is `O(k)` extra unembed projections per token, where `k` is small (typically 2–4). No memory overhead — nothing is cached.

## Why it matters

- **Free reasoning boost on any aligned model.** No fine-tuning, no architectural change, no extra memory. Drop-in at decode time.
- **Localises the "alignment tax."** Confirms the tax lives in the *final layers* and can be skipped — important for the literature on alignment-without-cost.
- **Generalises across dense and MoE.** Works on both architecture classes in the paper's experiments, suggesting the Guess–Refine–Perturb pattern is structural to alignment-tuned LLMs.

## Gotchas & tricks

- **Entropy threshold tuning matters.** Too tight → falls through to final layer too often. Too loose → accepts mid-layer logits that haven't finished refining.
- **Backward-search depth `k`.** Small `k` (2–4) captures the alignment-tax window; larger `k` risks reading off a layer that hasn't finished mid-stage refinement and tanks quality.
- **Does not work on base (non-aligned) models.** The whole pattern is *alignment-induced*. A base model's final layer is the right one.
- **Logit-lens-style projection** is required to read intermediate layers as logits; for non-canonical residual streams (some MoE variants) this can be noisy and require small per-layer calibration.

## Sources

- Paper: *Deeper is Not Always Better: Mitigating the Alignment Tax via Confident Layer Decoding* — Zhang, Zhoubian, Chen et al., Qwen (Alibaba) / Tsinghua / NTU, 2026 — [arXiv:2606.21906](https://arxiv.org/abs/2606.21906).
- Background: *Logit Lens* — nostalgebraist, 2020 — the projection technique used to read intermediate-layer logits.
