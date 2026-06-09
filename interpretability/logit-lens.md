# Logit lens
*Depth — project intermediate hidden states through the unembedding matrix to read off the model's "in-progress" token distribution.*

**TL;DR:** At any layer, multiply the residual-stream vector by the final unembedding matrix `W_U` (and softmax) to get a token distribution. The trajectory of these distributions across layers is a cheap, qualitative readout of how the model commits to a prediction. Cheap to implement, not always faithful, and the foundation for several behavioural-interp techniques.

**Prereqs:** [attention](../fundamentals/attention.md), [transformer-block](../architectures/transformer-block.md)
**Related:** [embedfilter](embedfilter.md), [causal-graph-explanations](causal-graph-explanations.md)

---

## What it is

A diagnostic, not a model edit. For a transformer with final unembedding `W_U` and (optionally) final-layer-norm parameters `LN_f`, define:

```
logit_lens(h_ℓ) = softmax(W_U · LN_f(h_ℓ))
```

for every layer `ℓ`. This treats the residual stream at layer `ℓ` *as if* the model were going to predict directly from it — a counterfactual readout of the model's "current best guess" at each depth.

## How it works

- Hook the residual stream after every transformer block.
- Apply the final layer norm (using the model's frozen `LN_f` parameters) and the unembedding.
- Softmax → token probabilities at every layer.

Variants:
- **Tuned lens** — learn a per-layer affine `A_ℓ` so that `softmax(W_U · LN_f(A_ℓ h_ℓ))` matches the final distribution as closely as possible. Reduces bias from the raw lens.
- **DLA (direct logit attribution)** — attribute the *logit difference* for a specific token to each transformer block by reading the unembedding alignment of its output.

## Why it matters

- Lets you watch the model "make up its mind" — early layers often hold lexical / surface features, middle layers contain abstract relations, late layers commit to one token.
- Underlies many behavioural-interp techniques (early-exit, model surgery, debugging hallucinations).
- The same `W_U` projection idea extends beyond logits: viewing `W_U` as a *feature lens* exposes that the residual stream uses frequent-token writing directions for purposes unrelated to predicting those tokens (see [EmbedFilter](embedfilter.md)).

## Gotchas & tricks

- **Faithfulness is suspect at early layers.** Raw lens distributions can look like noise or like the input. Use a tuned lens before drawing strong conclusions.
- **Models with weight-tied embed/unembed** (small GPT-2, some Llama variants) sometimes produce trivial lens outputs that just match the input token.
- **Layer norm matters.** Forgetting `LN_f` produces wildly miscalibrated distributions; always apply the final norm.
- **The lens is a probe, not a mechanism.** A token being "high under the lens" at layer 12 doesn't mean the model has decided on it — only that the residual stream happens to be aligned with that row of `W_U`.

## Sources

- Blog: *Interpreting GPT: the Logit Lens* — nostalgebraist, 2020 — first articulation of the technique.
- Paper: *Eliciting Latent Predictions from Transformers with the Tuned Lens* — Belrose et al., 2023 — quantitative critique and the tuned-lens fix.
- Paper: *Your UnEmbedding Matrix is Secretly a Feature Lens for Text Embeddings* — 2026 — [arXiv:2606.07502](https://arxiv.org/abs/2606.07502) — extends the lens idea from logits to embedding directions.
