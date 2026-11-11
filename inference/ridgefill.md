# RidgeFill
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A training-free prefill acceleration: run the model's **lower layers exactly** for the prompt, then use a **one-time-fit ridge regression** to predict the inputs to the upper layers and fill their KV caches analytically. Shortens prefill with small measured quality changes on two models — a quality-for-latency knob, not a free lunch.

**Prereqs:** [./README.md](./README.md), [./forkattention.md](./forkattention.md)
**Related:** [../architectures/mla.md](../architectures/mla.md)

---

## What it is

Prefill cost dominates time-to-first-token in long-context serving: all $N$ prompt tokens go through every layer before any decode step begins. Prior speedups either skip layers (quality loss) or cache across requests (not help for fresh prompts).

RidgeFill's observation: for a given model, the mapping from layer-$\ell$ activations to layer-$\ell'$ activations (for a late $\ell'$) is approximately linear *in expectation over prompts*. A ridge regression fit once, offline, lets you fill the upper layers' KV cache directly from the lower layers' outputs, without computing the intermediate layers.

## How it works

**Offline (one-time fit):**

1. Collect pairs $(h_\ell^{(i)}, h_{\ell'}^{(i)})$ of lower-layer and upper-layer activations across a calibration prompt set.
2. Fit a ridge regression $W_\text{predict}$ minimizing $\|W h_\ell - h_{\ell'}\|^2 + \lambda \|W\|^2$. Save $W_\text{predict}$ alongside the model.

**Online (prefill):**

1. Run lower layers $[1, \ell]$ exactly for every prompt token — standard transformer prefill, no change.
2. Apply $W_\text{predict}$ to get predicted inputs to layer $\ell'$, and use those inputs to compute KVs for layers $[\ell', L]$ — again as a standard matmul, no training.
3. Skip exact computation of layers $(\ell, \ell')$ entirely for prefill. Decode still runs all layers exactly.

The result is a filled KV cache for all layers; subsequent decode is standard.

## Why it matters

- **Training-free** — drops into any serving stack without reopening pretraining.
- **Prefill-only** — decode quality is untouched (decode uses the real layers); only the cache-fill is approximate. This constrains the quality cost to context grounding only.
- **Cousin of speculative decoding** — spec decoding trades decode compute for a drafter; RidgeFill trades prefill compute for a predictor. Both are quality-neutral in their reference region and quality-trading past it.

## Gotchas & tricks

- **Honest quality claims.** The paper explicitly flags that some confidence intervals include zero on multi-point quality losses — read as "small expected degradation, non-trivial variance", not "lossless".
- **Calibration set choice matters.** A narrow calibration set (e.g. only web text) hurts on code / long-context / multilingual prompts. Fit on a diverse corpus.
- **Choice of $\ell, \ell'$ is tuneable.** Early-to-late predictors save more compute but have more error; later-to-latest predictors save less and are more faithful. Usual choice: skip the middle third.
- **Interacts with context length.** Longer contexts have more opportunity for predictor error to compound; the paper reports per-length ablations.
- **Independent-researcher caveat.** The paper is a single-author independent submission; reproduction on larger models is still open.

## Sources

- Paper: *RidgeFill: Training-Free Late-Layer Cache Fill for Faster Prefill* — Robert P. Smith, 2026 — [arXiv:2611.09709](https://arxiv.org/abs/2611.09709)
