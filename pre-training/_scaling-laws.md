# Neural Scaling Laws

*Taxonomy — empirical functional forms predicting loss as a function of model size, data, and compute.*

**TL;DR:** A scaling law fits an analytic curve $L(N, D, C)$ to observed loss across model sizes $N$, dataset sizes $D$, and compute budgets $C$, so frontier labs can choose where to spend the next round of compute. The dominant families are **Kaplan-style** (single power law per axis), **Chinchilla-style** (joint $N$–$D$ fit with compute-optimal scaling), **Broken Neural Scaling Laws** (handle regime transitions / emergent breaks), and **Unified Neural Scaling Laws** (multi-axis, break-tolerant). Choice of functional form matters enormously for compute allocation decisions — Chinchilla famously reversed Kaplan's data/model ratio.

**Related taxonomies:** *(none yet)*
**Depth files covered here:** *(none yet — depth files for individual scaling laws will accumulate here)*

---

## The problem

Pretraining a frontier LLM costs $10^7$–$10^9$ dollars. The decision *how big a model, on how much data, with what compute* must be made before training starts. Scaling laws fit small-scale data ($10^7$–$10^{10}$ params) and *extrapolate* to the frontier ($10^{11}$–$10^{13}$ params) — when the extrapolation is wrong, you waste a budget.

What goes wrong with the wrong functional form:
- **Wrong asymptote** → underpredicts gains beyond a point; you stop training too early.
- **No support for regime breaks** → emergent capabilities or saturation kinks fit badly, hurting extrapolation.
- **Single-axis fits in a multi-axis world** → Kaplan fit $L(N)$ at fixed $D$; ignoring the joint $N$–$D$ tradeoff led to the famous "Kaplan's law is wrong" story when Chinchilla showed the optimum was much smaller models trained on much more data.

---

## The shared pattern

Every scaling law has:
- **A functional form** with a small number of free parameters (typically 3–8).
- **A loss target** — usually language-modeling cross-entropy on a held-out distribution.
- **A scaling parameter set** — $\{N, D\}$ minimum, often $\{N, D, C\}$ or with mixture / inference axes.
- **A fitting procedure** that estimates the parameters from observed training runs.
- **An extrapolation regime** beyond which the form is known or suspected to break.

Modern scaling laws also try to handle **breaks** — points where the curve's slope changes sharply (emergent capabilities, dataset exhaustion, architectural-budget transitions).

---

## Variants

| Form | Key idea | Main tradeoff | When it wins |
| --- | --- | --- | --- |
| **Kaplan** (Kaplan et al., 2020) | Independent power laws in $N$, $D$, $C$ with sharp regime boundaries | Single-axis; under-models the $N$–$D$ joint optimum | Early sizing decisions, single-axis sweeps |
| **Chinchilla** (Hoffmann et al., 2022) | Joint $L(N, D) = A/N^\alpha + B/D^\beta + E$; compute-optimal $N \propto C^{0.5}$, $D \propto C^{0.5}$ | Assumes no regime breaks; one form across all scales | Most pretraining-planning decisions through ~2024 |
| **Broken Neural Scaling Laws (BNSL)** (Caballero et al., 2023) | Piecewise power law with explicit "breaks" — captures emergent capabilities, double-descent, saturation | More parameters; harder to fit on sparse data | When emergence or regime change is suspected |
| **DeepSeek scaling laws** (DeepSeek 2024) | Chinchilla-style joint fit re-calibrated for the new data-quality regime; introduces *batch-size* and *learning-rate* scaling | Backend-specific (trained on cleaner data) | Modern open recipes; the de facto reference for 2024–25 |
| **Unified Neural Scaling Laws (UNSL)** (Caballero et al., 2026) | Multi-axis, break-tolerant — generalizes BNSL to arbitrary input axes simultaneously (e.g. $N \times D \times$ mixture $\times$ compute) | Most expressive; needs more data points to identify; risk of overfitting | Multi-axis sweeps; modern compute-allocation planning |
| **Inference-time scaling laws** (OpenAI o1 era, 2024) — *no depth file yet* | Loss / accuracy as a function of *inference-time compute* (CoT length, search budget) | Reverses the usual axis — model fixed, inference scales | Reasoning models, search-augmented inference |
| **Mixture / data-quality scaling laws** (various, 2024–) — *no depth file yet* | Adds dataset *composition* as a scaling axis (web vs code vs synthetic) | Hard to enumerate axes; high-dimensional fit | Modern data-curation planning |

---

## How to choose

**Default for new pretraining planning:** Chinchilla as a baseline + DeepSeek's recalibration if you have clean data. The compute-optimal $N \propto C^{0.5}$ rule is the right starting point.

**If you suspect emergence or a regime break:** fit BNSL. Don't extrapolate Chinchilla across a known break.

**If you're co-varying multiple axes** (model size × data × mixture × inference-compute): UNSL is the only form that fits multi-axis breaks without piecewise stitching.

**If your axis is inference compute** (reasoning, search budget): the inference-time scaling literature is its own thing — don't try to reuse pretraining functional forms.

**Common mistake:** fitting a power law to noisy small-scale data and extrapolating two orders of magnitude. All functional forms break eventually; the question is whether the break is *within* your extrapolation window. Always validate at the largest scale you can afford before committing the rest of the budget.

---

## Adjacent but distinct

- **Compute-optimal training schedules** ([wsd-schedule](wsd-schedule.md), [_lr-schedules](_lr-schedules.md)) — operationalize a scaling law's prescription but aren't themselves scaling laws.
- **Architectural-efficiency comparisons** (FLOP-matched A vs B) — single-scale, doesn't extrapolate. Useful but different question.
- **Emergent-capability literature** (Wei et al. 2022, "Emergent Abilities…") — what BNSL / UNSL break-modeling is trying to capture quantitatively.
- **Inference-time scaling for reasoning** — distinct axis (compute spent at *test* time, not *train* time); see Snell et al. 2024, the o1 system card.

---

## Sources

- Paper: *Scaling Laws for Neural Language Models* — Kaplan et al., OpenAI, 2020.
- Paper: *Training Compute-Optimal Large Language Models (Chinchilla)* — Hoffmann et al., DeepMind, 2022.
- Paper: *Broken Neural Scaling Laws* — Caballero, Gupta, Rish, Krueger, ICLR 2023.
- Paper: *DeepSeek LLM: Scaling Open-Source Language Models with Longtermism* — DeepSeek, 2024.
- Paper: *Unified Neural Scaling Laws* — Caballero, Jaini, Krueger, Rish, 2026 — [arXiv:2605.26248](https://arxiv.org/abs/2605.26248).
