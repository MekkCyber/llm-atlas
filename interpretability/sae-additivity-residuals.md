# SAE Additivity Residuals (SAR)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** SAEs decode linearly, but the **behavioral** effects of their features need not compose additively. SAR measures the behavioral variation that remains *unexplained* above a chosen interaction order — a direct quantification of how additive your SAE really is. Under independent feature-subset sampling, population SAR equals the normalized energy of interactions above that order.

**Prereqs:** [sae.md](./sae.md)
**Related:** none yet

---

## What it is

The hope: *behavior*$(\sum_{i\in S} f_i) \approx \sum_{i\in S} g_i$ for some per-feature behavioral effects $g_i$. If this holds, SAE features compose additively and you can interpret or steer each feature independently.

The reality: the downstream computation after the SAE site is non-linear, so additivity is at best an approximation. SAR formalizes "how far off" by measuring the energy in interaction terms above a chosen order.

Operationally, treat behavior as a function of a *coalition* $S \subseteq \{1, \ldots, N\}$ of active SAE features. Any such coalition function decomposes into a Möbius / Hoeffding sum:

$$
B(S) = \sum_{T \subseteq S} \phi(T) \qquad \phi(T)\ \text{is the interaction term of order}\ |T|
$$

The order-$k$ **truncation** $B_k(S) = \sum_{T \subseteq S,\ |T|\le k} \phi(T)$ is the best additive-plus-interactions-up-to-$k$ approximation.

## How it works

### Population SAR

Under **independent feature-subset sampling** (each feature's presence in the coalition is independent with some rate $p$), population SAR of order $k$ is

$$
\mathrm{SAR}_k = \frac{\|B - B_k\|^2_{\text{sample}}}{\|B\|^2_{\text{sample}}}
= \frac{\sum_{|T| > k} \phi(T)^2 \cdot w(T)}{\sum_{T} \phi(T)^2 \cdot w(T)}
$$

That is, the normalized **energy of interactions above order $k$**. It bounds the ordering error of the optimal low-order approximation on held-out coalitions, so it's not just a goodness-of-fit statistic — it bounds *decisions* based on the approximation.

### Learned coordinates vs sum-preserving random rotations

To separate compositionality coming from the *representation* (the SAE directions) from compositionality in the *downstream computation*, SAR compares:

- **Learned SAE coordinates** — the actual features.
- **Sum-preserving random rotations** of the same selected subspace — directions with the same total energy but no particular alignment to behavior.

If first-order residuals ($\mathrm{SAR}_1$) are **lower** for the learned coordinates, the SAE is doing representational work the downstream computation can't do by itself.

### Combined-feature steering experiment

Beyond measurement, SAR gives a drop-in **pairwise upgrade** for feature steering: when steering with multiple features, add pairwise interaction terms to the predicted effect. Across 18 model/behavior settings, the paper reports rank fidelity 0.5874 → **0.7790** and ordering error 0.1444 → **0.0606** after adding pairwise terms.

## Why it matters

- **Settles a background debate.** "Are SAE features compositional?" wasn't well-defined. SAR gives a measurable, falsifiable version with closed-form interpretation.
- **Both results, both useful.** (a) Learned coordinates beat random rotations on first-order fit, so SAEs really do capture additive structure. (b) Pairwise terms improve every setting, so faithful joint intervention needs more than first-order.
- **Actionable for steering.** Any pipeline doing multi-feature SAE steering should run with pairwise terms, with a quantified lift.

## Gotchas & tricks

- **Independent subset sampling is an assumption.** Natural feature co-occurrence in real prompts is correlated; SAR under correlated sampling needs a different estimator (importance correction).
- **Order $k$ truncation ≠ Shapley.** SAR measures the squared-error loss from truncating the Möbius expansion; Shapley measures marginal contributions. They coincide only under special weights.
- **Pairwise is not a silver bullet.** The paper notes some settings still have substantial $\mathrm{SAR}_2$ — higher-order terms matter for complete faithfulness. Pairwise is a strict improvement, not a resolution.
- **Random rotations must preserve the sum.** Any orthogonal transform of the SAE's selected subspace preserves total feature energy; the comparison is only meaningful inside such transforms.
- **Scope.** 18 model/behavior settings, including steering experiments. Scaling to large open models or very-high-order interactions is open.

## Sources

- Paper: *Behavioral Compositionality of Sparse Autoencoder Representations* — Ziye Mei, Frederic Sala (University of Wisconsin–Madison), 2026 — [arXiv:2610.21261](https://arxiv.org/abs/2610.21261).
- Code: https://github.com/OpenSAE/SAE-Behavioral-Compositionality
