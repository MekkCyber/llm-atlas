# Influence Matching (Inf-Match) for Dataset Distillation
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Dataset distillation typically matches *process surrogates* — per-step gradients or full training trajectories — between synthetic and real data. Inf-Match matches the **outcome** instead: it learns a small synthetic set whose *influence on the converged model parameters* matches the real dataset's. The core ingredient is a fully differentiable, linear-time, sample-level **influence estimator** built by unrolling optimizer dynamics and applying a first-order Taylor expansion — no inverse-Hessian products, no convexity assumptions.

**Prereqs:** [_data-curation.md](./_data-curation.md), [quality-filtering.md](./quality-filtering.md).
**Related:** [../interpretability/README.md](../interpretability/README.md)

---

## What it is

Dataset distillation compresses a large training set $D$ into a tiny synthetic set $\tilde{D}$ (say 10 images per class) such that a model trained on $\tilde{D}$ approximates one trained on $D$. Prior methods (gradient matching, trajectory matching, NCFM) align the *process* — instantaneous gradients or step-wise trajectories — under the assumption that matching the process implies matching the outcome. In practice the assumption is weak: matched trajectories often diverge in final parameters, and matched final parameters can be reached along wildly different trajectories.

Inf-Match aligns the *outcome* by defining the alignment target as **influence on converged parameters**:

$$
\mathcal{L}_{\text{Inf-Match}} = \big\| \mathcal{I}(D) - \mathcal{I}(\tilde{D}) \big\|_2^2
$$

where $\mathcal{I}(S)$ measures how much adding/removing $S$ shifts the trained parameters.

## How it works

The bottleneck is $\mathcal{I}$: classical influence functions (Koh & Liang, 2017) require inverse Hessian-vector products, which are infeasible at modern scale and assume convexity that neural training doesn't satisfy.

Inf-Match's estimator:

1. **Unroll the optimizer for $T$ steps** on the real dataset, recording state trajectory $\theta_0, \dots, \theta_T$.
2. **Perturb the training data** by adding/removing a small set $S$ and compute the induced trajectory perturbation to first order (Taylor expansion around the unperturbed trajectory).
3. **Aggregate** the per-sample parameter shift as $\mathcal{I}(S)$.

The result is fully differentiable in the synthetic set $\tilde{D}$ (the perturbation), runs in $O(n)$ per sample, and requires neither inverse Hessians nor convexity. Training the synthetic set is then a straightforward gradient descent on $\mathcal{L}_{\text{Inf-Match}}$.

## Why it matters

- **Outcome-aligned target** is what practitioners actually want: they care about the accuracy of a model trained on $\tilde{D}$, not whether its gradients momentarily agreed with the full set's.
- **Practical scale**: linear-time influence removes the biggest blocker to using influence-function-style tools at modern model sizes.
- **Beyond classification**: Inf-Match scales to vision-language distillation (Flickr30K image/text retrieval), a domain earlier process-matching methods struggled with.
- The **differentiable influence estimator is useful on its own** — it's a general primitive for data selection, curriculum design, and interpretability (identifying training examples most responsible for a model's behavior on a query).

Reported numbers: Tiny-ImageNet IPC=10 → 31.5% (+4.7 vs NCFM); Flickr30K retrieval average +2.5 vs NCFM at 200–1000 synthetic samples.

## Gotchas & tricks

- **First-order Taylor is only local.** For synthetic sets that induce large perturbations from the real-data trajectory, the estimator's accuracy degrades. Small $\tilde{D}$ (the standard distillation regime) is where it's most trustworthy.
- **Unroll length $T$**. Longer unrolls give more accurate influence but multiply memory. Paper reports the sweet spot at moderate $T$.
- **Not just for images.** The estimator is architecture-agnostic; the win on Flickr30K vision-language distillation suggests it generalizes.
- **Distinct from coreset selection.** Coresets pick *real* samples that summarize $D$; dataset distillation learns *synthetic* samples that outperform any real subset of the same size.

## Sources

- Paper: *Dataset Distillation by Influence Matching* — Haoru Tan et al. — HKU / CUHK / Stanford, 2026 — [arXiv:2607.16859](https://arxiv.org/abs/2607.16859)
- Code: [github.com/hrtan/infmatch](https://github.com/hrtan/infmatch)
- Related: Koh & Liang (2017) *Understanding Black-box Predictions via Influence Functions*.
