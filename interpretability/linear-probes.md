# Linear Probes
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Fit a linear classifier on top of frozen hidden states to predict whether a concept (sentiment, truth, refusal, a feature) is present. The probe's weight vector is the "direction" that best detects the concept. Classic interpretability tool, but its **best-detection direction is not the best steering direction** — a result formalised by Sadeghi & Becker (2026).

**Prereqs:** [../fundamentals/attention.md](../fundamentals/attention.md), [../architectures/transformer-block.md](../architectures/transformer-block.md)
**Related:** [activation-steering.md](activation-steering.md), [self-editing-llms.md](self-editing-llms.md)

---

## What it is

A supervised-classification tool on top of model internals. Given paired activations $\{(h_i, y_i)\}$ at some layer $\ell$ — $y_i \in \{0, 1\}$ labels whether the concept is present — fit a logistic regression or linear SVM. The learned weight vector $w$ points along the concept's discriminative direction; the probe's accuracy measures how linearly separable the concept is at that layer.

## How it works

1. **Probe set.** Collect paired inputs (concept-present vs concept-absent) and capture hidden states at one or more layers.
2. **Fit.** $\hat{w} = \arg\min_w \sum_i \ell(y_i, w^\top h_i)$. Regularisation (L2, mean-pooling) matters for noisy settings.
3. **Interpret.** High accuracy at a given layer ⇒ the concept is linearly readable there. The direction $w$ is a candidate "concept vector" for downstream use (steering, interpretability, safety eval).

### Variants

- **Difference-of-means (MD-pool).** Not a probe in the classifier sense — just $v = \mathrm{mean}(h^+) - \mathrm{mean}(h^-)$. Related but distinct direction; see Gotchas.
- **Multi-token probes.** Pool across a token window instead of a single position.
- **Probing with a cross-model linear map.** Train a linear map $M$ between two models and check how well probe direction $M w$ works in the target.

## Why it matters

- **Backbone of the "linear representation hypothesis"** that modern interp operates under.
- **Fast, low-compute way** to check whether a safety-relevant concept (deception, refusal, situational awareness) is explicit in the hidden state.
- **Cross-model transferable (ish).** Sadeghi & Becker report that a linearly-mapped probe keeps most of its accuracy across models, while a linearly-mapped steering vector keeps much less.

## Gotchas & tricks

- **Probe ≠ steer.** The direction with the best classification accuracy is usually not the direction that steers the model best — the two optimise different objectives.
- **Layer sweeps matter.** Probes often plateau mid-depth and regress near the output; report the full curve.
- **Dataset artefact risk.** Easy concepts can be read off from format tokens or positional artefacts; always ablate the trivial baseline.
- **Linear separability ≠ causal use.** A concept can be probed accurately and still not be the direction the model actually uses for its decision. Causal interventions (patching, activation steering) are needed to close that loop.
- **MD-pool vs trained probe.** MD-pool is cheap, class-balance-dependent, and often *closer to the best steering direction*. Trained probes are better for detection but worse for intervention.

## Sources

- Paper: *Discovering Latent Knowledge in Language Models Without Supervision* — Burns et al., 2022 — CCS / unsupervised probes.
- Paper: *Linear Representations of Sentiment in LLMs* — Tigges et al., 2023.
- Paper: *Do Probing and Steering Favor the Same Directions Within and Across Language Models?* — Sadeghi & Becker, 2026 — the probe≠steer gap, measured across three instruction-tuned LMs and under linear cross-model maps.
