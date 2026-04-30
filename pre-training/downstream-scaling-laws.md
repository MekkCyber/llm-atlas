# Downstream-Task Scaling Laws (Llama 3's Extension)

*Depth — predict benchmark accuracy from training compute, not just loss.*

**TL;DR:** Chinchilla predicts validation loss as a function of (N, D). But practitioners care about **benchmark accuracy** (MMLU, ARC, HumanEval), not loss. Llama 3 (Sec. 3.2.1) extends Chinchilla-style fits with a two-stage pipeline: **compute → normalized NLL per character (linear fit) → benchmark accuracy (sigmoidal fit)**. Anchored with the Llama 2 family at larger compute. Extrapolates over 4 orders of magnitude in compute to predict 405B's ARC-Challenge score — only slightly under the measured result. The first concrete recipe for "what accuracy does C FLOPs buy me on benchmark X."

**Prereqs:** [chinchilla-scaling](chinchilla-scaling.md)
**Related:** [annealing-as-data-eval](annealing-as-data-eval.md) · [llama-3 case study](../case-studies/llama-3.md)

---

## What it is

Chinchilla's compute-to-loss map (`C → L`) is useful for deciding N and D but doesn't tell you "will my model be good at MATH?". There's no closed-form derivation from loss to benchmark accuracy — the relationship depends on the benchmark and the model's capability profile.

Llama 3's contribution: a **two-stage fit** that maps compute directly to benchmark accuracy, using the IsoFLOPs data from Chinchilla-style runs plus older-model anchors for extrapolation.

---

## How it works

### Stage 1 — Compute → Normalized NLL per character

Run IsoFLOPs grid (Sec. 3.2.1):
- Compute budgets: `6 × 10¹⁸ → 10²²` FLOPs.
- Model sizes: `40M → 16B` parameters.
- Cosine LR + 2,000-step warmup.
- Peak LR `2 × 10⁻⁴ to 4 × 10⁻⁴` depending on size.
- Weight decay = `0.1 × LR_t`.
- Batch size scaled with compute, 250K to 4M tokens.

For each compute budget, identify the **compute-optimal model** (lowest loss on the IsoFLOP curve). Evaluate that model's NLL on a downstream benchmark (e.g., ARC-Challenge):

```
normalized_NLL = NLL_of_correct_answer / length_of_correct_answer
```

Normalizing by length makes the NLL comparable across questions.

Plot `normalized_NLL` vs `log(compute)`. Empirically: **linear** (within the scaling-law model range).

```
normalized_NLL ≈ a - b · log₁₀(C)
```

Fit the linear coefficients.

### Stage 2 — NLL → Accuracy

Accuracy isn't a smooth function of NLL — it has a sigmoid-like shape: at high NLL, the model is random (accuracy = 1/K for K choices); at low NLL, accuracy saturates near 100%.

```
accuracy(NLL) = σ(a' - b' · NLL)
```

where σ is the logistic function.

**Anchor with larger-compute models.** The IsoFLOPs models top out at 16B × 10²² FLOPs. To extrapolate to 405B × 4 × 10²⁵ FLOPs, you need anchors at larger compute.

Llama 3 uses the **Llama 2 family** (7B, 13B, 34B, 70B) as NLL-accuracy anchors. Even though Llama 2 has a different tokenizer and data mix, the *sigmoid shape mapping NLL to accuracy* is robust enough to use across lineages.

### Putting it together

Chain the two fits:

```
compute C → (stage 1) → predicted NLL(C) → (stage 2) → predicted accuracy(C)
```

For 405B at 3.8 × 10²⁵ FLOPs:
- Stage 1 predicts NLL ≈ some value (extrapolated from 10²² FLOPs IsoFLOPs data).
- Stage 2 converts that NLL to accuracy via the sigmoid fit.

Llama 3 reports (Sec. 3.2.1 Figure 4): on **ARC Challenge**, the prediction "slightly underestimates" the flagship Llama 3's actual accuracy. Given the extrapolation is over 4 orders of magnitude in compute, "slightly under" is remarkable fidelity.

### The full equation chain

No closed-form `accuracy(C)` formula is printed in the paper; it's the composition of two fitted maps:

```
compute = C
nll = a₁ - b₁ · log₁₀(C)                      (stage 1, linear)
accuracy = 1 / (1 + exp(-(a₂ - b₂ · nll)))    (stage 2, sigmoidal)
```

Coefficients `a₁, b₁, a₂, b₂` are per-benchmark (ARC has different coefficients than MMLU).

---

## Why it matters

- **Plan for accuracy, not loss.** Given a compute budget, you can predict "this will give me 75% on MMLU." That's what practitioners want to know, not "this will give me a loss of 1.98."
- **Rigorously anchored.** Using a scaling-law IsoFLOPs grid for stage 1 and a separate model family for stage 2 keeps the fits tied to actual measurements over 4 orders of magnitude.
- **Generalizes the Chinchilla framework.** Chinchilla → loss. Llama 3 → loss → accuracy. The two stages are modular: you can swap benchmarks or anchor families.
- **Fundamentally empirical.** No theorem; the shape of the fits is whatever the data shows. This is fine — it's what practitioners reach for.

---

## Gotchas & tricks

- **Per-benchmark, not universal.** The (a₁, b₁, a₂, b₂) constants differ across benchmarks. Fit each separately.
- **The sigmoid's "saturation" regime is a floor.** When accuracy > 95%, small NLL changes don't move accuracy much. Below 30%, NLL changes don't help either (model below random → random). Most of the discrimination is in the 40–80% regime.
- **Tokenizer mismatches bias the NLL-to-accuracy sigmoid.** When using Llama 2 anchors (different tokenizer) for Llama 3 predictions, the NLL-per-character unit is preserved across tokenizers only approximately. Llama 3 acknowledges this and says the fit works well empirically despite the mismatch.
- **Works best when the IsoFLOPs data is compute-optimal at each C.** If you sweep model sizes but don't find the optimum for each C, stage 1's NLL points are noisy.
- **Extrapolation is the hard part.** 4 orders of magnitude is a huge stretch. The fact that the 405B prediction is close (not exact) is evidence the functional form is reasonable for this range of compute — but there's no guarantee it extrapolates to 10²⁸ FLOPs.
- **Saturated benchmarks tell you less.** MMLU saturates near 89% (human expert estimate). Predicting "this model gets 87% vs 89%" is within the fit's noise floor.
- **Different data mixes have different stage 1 fits.** Changing your pretraining data mix changes the NLL-vs-compute relationship. Re-fit when the mix changes significantly.
- **Can't predict emergence.** If a benchmark has a phase transition (sudden capability "emergence" at some scale), the sigmoid fit may miss it. Chain-of-thought, arithmetic, multi-step reasoning are examples of benchmarks with emergent-like curves.

---

## Relation to annealing-as-eval

Llama 3 uses annealing (see [annealing-as-data-eval](annealing-as-data-eval.md)) to **evaluate candidate data sources** at small scale and predict their value at large scale. Complementary to this page's technique:

- **Downstream scaling laws**: given a fixed data mix, predict accuracy-vs-compute.
- **Annealing-as-eval**: given a candidate data source, predict its marginal value via a cheap anneal on a small model.

Both are "predict large-scale behavior from small-scale experiments." Both are Llama 3 contributions.

---

## Sources

- Paper: *The Llama 3 Herd of Models* — Meta, 2024, arXiv 2407.21783, Sec. 3.2.1 — introduces the two-stage downstream scaling fit.
- Paper: *Training Compute-Optimal Large Language Models* — Hoffmann et al., 2022, arXiv 2203.15556 — [chinchilla-scaling](chinchilla-scaling.md), the prerequisite compute-to-loss framework.
- Paper: *Predicting Emergent Capabilities by Finetuning* — Dubey et al., 2023, arXiv — prior work on prediction at scale.
