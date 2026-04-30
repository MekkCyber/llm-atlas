# Annealing as a Data-Evaluation Tool

*Depth — Llama 3's trick for predicting the value of a candidate dataset cheaply.*

**TL;DR:** Instead of running full scaling-law sweeps for every candidate dataset, **anneal** a half-trained small model on a mix of (30% candidate dataset + 70% default mix) over **40B tokens** with LR linearly decaying to zero. Measure benchmark improvements. The anneal's effect approximates the candidate's marginal value at larger scale. Llama 3 reports **+24% GSM8k, +6.4% MATH on 8B** via this method — substantial, at a fraction of the cost of a full scaling-law run (Sec. 3.1.3). Specific to Llama 3; broadly reusable pattern.

**Prereqs:** [chinchilla-scaling](chinchilla-scaling.md), [_data-curation](../data/_data-curation.md)
**Related:** [data-mix](../data/data-mix.md) · [downstream-scaling-laws](downstream-scaling-laws.md)

---

## What it is

A cheap proxy evaluation for candidate pretraining data sources. Instead of:

1. ❌ **Full scaling-law sweep**: train N small models with vs without the candidate data; fit scaling laws; extrapolate → expensive (multiple full pretraining runs per candidate).
2. ❌ **Post-hoc benchmarking**: train one large model, benchmark, iterate → catastrophically slow.

Use:

3. ✅ **Annealing evaluation**: take a partially-trained model (50% through pretraining), anneal LR → 0 on a mix with the candidate dataset upsampled → measure benchmark delta.

Llama 3 (Sec. 3.1.3) explicitly describes this as a methodology:

> *"We measure the value of such datasets by annealing the learning rate of a 50% trained Llama 3 8B model linearly to 0 on 40B tokens. In those experiments, we assign 30% weight to the new dataset and the remaining 70% weight to the default data mix. Using annealing to evaluate new data sources is more efficient than performing scaling law experiments for every small dataset."*

---

## How it works

### The evaluation recipe

```
Start: 50%-trained Llama 3 8B  (midway through pretraining; optimizer state and LR mid-run)
    ↓
Anneal phase:
    - data mix: 30% candidate dataset + 70% default mix
    - 40B tokens
    - LR linearly decays from current rate to 0 over the 40B
    ↓
Final checkpoint → benchmark evaluation
    ↓
Delta vs baseline anneal (100% default mix over same 40B): candidate's marginal value
```

Why this works:
- **Upsampling to 30%** ensures the candidate is well-represented during the anneal (not drowned in the default mix).
- **LR → 0** means the anneal weights converge and benchmark numbers are stable (not mid-training fluctuation).
- **Half-trained start** means the model has enough general capability to benefit from the candidate data but is still plastic enough to respond.
- **40B tokens** is large enough for the anneal to converge but small enough to be cheap.

### The benchmark as the signal

Measure on downstream benchmarks (GSM8K, MATH, HumanEval, MMLU, etc.). The delta vs a baseline anneal (no candidate upsampling) quantifies the candidate's value.

Reported Llama 3 deltas (Sec. 3.1.3) on an 8B model:
- **GSM8K: +24.0%**
- **MATH: +6.4%**

Large signal — clearly worth including this candidate. For weaker candidates, deltas closer to 0; discard.

### When it doesn't work

Llama 3 notes (Sec. 3.1.3): *"Improvements on the 405B model are negligible, suggesting that our flagship model has strong in-context learning and reasoning capabilities and does not require specific in-domain training samples."*

The 8B model benefits; the 405B doesn't (the flagship's in-context capability makes the domain-data lift marginal at large scale). So annealing-as-eval **works best on small models**; deltas predict small-model ceiling improvements, not large-model ones.

For the main-run 405B's final anneal, Llama 3 used 40M tokens (not 40B) of the pre-selected high-quality data.

### Generalizes to any candidate

- New web corpus → anneal, measure delta.
- New code dataset → anneal, measure delta.
- New low-resource-language corpus → anneal, measure per-language perplexity.
- New synthetic dataset (math problems, code exercises) → anneal, measure.

Turnaround time: ~hours on a small 8B (annealing 40B tokens on a few GPUs).

---

## Why it matters

- **Fast iteration on data.** Before this, evaluating a candidate dataset required a full pretraining run (days, many GPUs). Now it's hours on a smaller setup.
- **Decouples data from architecture decisions.** You can iterate on data sources rapidly without retraining the base model.
- **Formalizes the "try it and see" approach.** Most labs probably did something like this informally; Llama 3 publishes the specific recipe.
- **Complements scaling-law experiments.** For new domains (e.g., adding a new language), you can anneal-eval first, then commit to a full scaling sweep only if the anneal shows promise.
- **Reusable methodology.** Open-recipe models (OLMo 2, Gemma 2) likely use similar patterns internally.

---

## Gotchas & tricks

- **50%-trained checkpoint is the sweet spot.** Earlier checkpoints are less capable (deltas are noisy); later checkpoints are less plastic (deltas are smaller).
- **LR decay to zero matters.** Without full decay, the anneal isn't converged; benchmark numbers fluctuate.
- **Evaluation benchmark choice.** Anneal-evaluate with diverse benchmarks; some candidate data may help one benchmark and hurt another.
- **Candidate weight 30% is a chosen value.** Too low (e.g., 10%) → weak signal; too high (e.g., 70%) → candidate drowns the default mix, unrealistic training distribution.
- **40B tokens is the scale.** Shorter (10B) gives weaker signal; longer (200B) is closer to a full scaling-law experiment.
- **Requires a pretrained checkpoint at 50%.** You need to have already committed to the main pretraining run. Adjusting pretraining plans mid-run based on anneal-eval results takes managerial coordination.
- **Small-model deltas don't always transfer.** The "negligible on 405B" finding means you can't use 8B anneal deltas to predict 405B outcomes reliably. Anneal-eval is for ceiling-improving small models.
- **Not a full scaling law replacement.** For "should I train a 70B on this data?" — you still need a bigger eval or a full scaling sweep. Anneal-eval is for "is this candidate worth including at all?"
- **Data quality bias.** Anneal-eval measures short-term benchmark lift. A dataset that boosts GSM8K but introduces adversarial patterns (e.g., over-fitting to test-style questions) looks good in anneal-eval but may hurt downstream.
- **Avoid evaluation-set contamination.** Candidate datasets should be decontaminated against the benchmark test sets. Otherwise the anneal-eval delta reflects memorization, not capability.

---

## Sources

- Paper: *The Llama 3 Herd of Models* — Meta, 2024, arXiv 2407.21783, Sec. 3.1.3 — the annealing-as-data-eval methodology.
- Paper: *Training Compute-Optimal Large Language Models (Chinchilla)* — Hoffmann et al., 2022 — the scaling-law baseline anneal-eval is a cheap proxy for.
- Paper: *WSD: Warmup-Stable-Decay* — Hu et al., 2024 — related annealing-style schedules.
- Related: [data-mix](../data/data-mix.md) for how these signals get composed into the actual training mix.
