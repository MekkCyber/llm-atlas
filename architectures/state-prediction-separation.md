# State-Prediction Separation
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A Transformer variant that splits the residual stream into **two computation streams** — one dedicated to shaping the next-token distribution ("prediction"), one dedicated to carrying information forward for future positions ("state"). Standard Transformers overload one stream with both roles; separating them consistently improves data and compute efficiency and yields **2–3 pp average uplift** on downstream tasks at every scale tested.

**Prereqs:** [transformer-block](transformer-block.md), [multi-head-attention](multi-head-attention.md)
**Related:** [reordered-norm](reordered-norm.md), [qk-norm](qk-norm.md), [../interpretability/README.md](../interpretability/README.md)

---

## What it is

In a standard Transformer, at every position $t$ the residual stream must simultaneously:

1. **Predict**: shape the logits for the next token, via the LM head.
2. **Store state**: carry information forward so later positions can attend to it.

These two roles pull the same vector in different directions — mech-interp work has long observed that the residual stream acts as a shared bus. **State-prediction separation** materialises the bus as two parallel streams with distinct roles: only the prediction stream feeds the LM head, only the state stream is read by attention at future positions.

## How it works

The transformer block is duplicated into a prediction path $h^p_t$ and a state path $h^s_t$. Both streams share the parameters where they can, but:

- The LM head reads $h^p_t$ only.
- Future positions' attention reads $h^s_{<t}$ only (keys and values are computed from $h^s$).
- Query is computed from $h^p$, so a token *asks* about the past through its prediction role but *becomes* past through its state role.

MLPs and attention residuals write to both streams (with per-stream projections). The extra parameters are a fixed multiplier over the baseline block, but the compute stays linear because the two streams travel through the same attention pattern.

## Why it matters

- **Free efficiency**: architectural, not systems-level — no new kernels, no all-to-all cost, and the block is drop-in for HF-style Transformers.
- **Consistent scaling**: gains reported across multiple scales, mirroring how GQA / MLA landed.
- **Mech-interp handshake**: the arch operationalises a mech-interp intuition (shared-bus overload) into a training-time architectural lever.

## Gotchas & tricks

- Parameter cost: doubling stream dimensions adds parameters. Reported gains come from keeping *width* fixed and paying with a modest FLOP overhead, not from widening both streams.
- The split only helps if the two roles were actually competing — for very small models the residual stream isn't saturated and the split can be neutral.
- Interactions with normalisation matter: reordered-norm-style pre/post-LN placement has to be redone per-stream; naïve copy of the baseline recipe loses the gain.
- Inference: KV cache is only for the *state* stream, so cache size is unchanged.

## Sources

- Paper: *The State-Prediction Separation Hypothesis* — Angelini, Ng, Artzi, Brantley, 2026 — [arXiv:2607.01218](https://arxiv.org/abs/2607.01218).
