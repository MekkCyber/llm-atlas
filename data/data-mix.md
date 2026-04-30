# Pretraining Data Mix

*Depth — how modern frontier models compose their pretraining corpus across domains.*

**TL;DR:** The composition of the pretraining corpus by domain (general knowledge, code, math, multilingual) is a load-bearing design choice, tuned via **scaling-law experiments over data mixes**. Llama 3's final mix (Sec. 3.1.2): **~50% general knowledge, ~25% math & reasoning, ~17% code, ~8% multilingual**. Picked via small-scale proxy runs that predicted large-scale outcomes. Also: **knowledge classification** for downsampling over-represented categories (arts/entertainment), and **mid-run mix adjustments** (upsample math, add recent data near the knowledge cutoff). The data mix is the highest-leverage pretraining lever outside of scale itself.

**Prereqs:** [chinchilla-scaling](../pre-training/chinchilla-scaling.md), [_data-curation](_data-curation.md)
**Related:** [dolma](dolma.md) · [quality-filtering](quality-filtering.md) · [annealing-as-data-eval](../pre-training/annealing-as-data-eval.md)

---

## What it is

The **proportional composition** of the pretraining corpus by domain. Models train on trillions of tokens drawn from many sources (web, code, math, textbooks, Wikipedia, papers); the mix determines how many tokens come from each.

Mix choice affects:
- **Capability profile**: more code → better code benchmarks; more math → better math.
- **Trade-offs**: increase one domain → decrease another. Not all domains have unlimited data.
- **Scaling-law behavior**: different mixes have different compute-to-loss curves.
- **Transfer**: some domains (code, math) transfer to general reasoning; some (entertainment trivia) don't.

---

## How it works

### Llama 3's mix (Sec. 3.1.2)

| Domain | Share |
|---|---|
| General knowledge | ~50% |
| Mathematical & reasoning | ~25% |
| Code | ~17% |
| Multilingual | ~8% |

Total: 15T tokens across all domains for Llama 3.

### How mixes are picked

The paper describes **scaling-law experiments over data mixes**:

1. Train a set of small models (e.g., 8B × different mixes) with different data proportions.
2. Measure their benchmark performance and loss.
3. Fit a scaling law over the (mix, scale) → performance relationship.
4. Extrapolate to choose the mix that maximizes predicted 405B performance.

This is "data-mix-as-hyperparameter" — treat the mix proportions as tunable parameters and sweep them at small scale to pick the large-scale mix.

### Knowledge classification (Sec. 3.1.2)

A secondary tool: classify documents by topic (arts/entertainment, history, science, ...) and **downsample over-represented topics**. The web is dominated by entertainment content; naively mixing in CommonCrawl proportions would give 50%+ arts/entertainment, which doesn't help much for reasoning.

Classification method: a trained classifier (likely another DistilRoBERTa) predicts topic; per-topic sampling weights rebalance.

### Mid-run adjustments (Sec. 3.4.1)

Llama 3 adjusted the data mix **during training**:
- **Upsampled non-English** as training progressed.
- **Upsampled math**.
- **Added recent web data later** (to extend the knowledge cutoff without polluting early training with recency bias).
- **Downsampled low-quality subsets** identified later.

Mid-run adjustment is rare in the open literature but apparently common internally at large labs. Requires careful validation to avoid introducing instabilities.

### Annealing data as final-stage adjustment (Sec. 3.1.3, 3.4.3)

At the very end of pretraining, Llama 3 **anneals** on a **high-quality mixture** that upsamples specific domains (math, code, knowledge-intensive). See [annealing-as-data-eval](../pre-training/annealing-as-data-eval.md).

The annealing mix is distinct from the main mix: narrower, higher quality, for the last ~40M tokens at LR decay to zero.

---

## Why it matters

- **Highest-leverage pretraining lever besides scale.** Chinchilla tells you N and D; data mix tells you *which* D. Directly affects downstream capability.
- **Where frontier-model differentiation lives.** Chat GPT vs Claude vs Llama differ as much in data mix as in architecture. The mix embeds the lab's priorities.
- **Scales with scaling-law experiments.** The methodology for picking mixes is now mature: small-scale sweeps + extrapolation.
- **Connects pretraining to post-training.** A model pretrained with 30% code (vs 10%) needs less code-specific SFT. Mix shapes the downstream pipeline.

---

## Gotchas & tricks

- **Data abundance varies by domain.** Web is ~unlimited; high-quality math is scarce; multilingual is finite per-language. Your mix ceiling is bounded by what exists.
- **Mix choice affects token count too.** If you allocate 25% to math, and you only have 3.75T math tokens available at quality, you're epoch-constrained on math and can't push to 15T total without epoch-reuse or quality degradation.
- **Synthetic data fills gaps.** Low-resource domains (multilingual, specialized math) can be augmented with synthetic data. Tradeoff with quality.
- **Scaling laws over mixes are expensive.** Each "mix candidate" is a full training run. Labs use cheap proxies (smaller models, shorter runs, cheaper benchmarks) to explore the space.
- **Evaluation matters for mix choice.** Optimizing for loss may not optimize for downstream benchmarks. Use benchmark-level scaling laws (Llama 3's [downstream-scaling-laws](../pre-training/downstream-scaling-laws.md)) to pick.
- **Knowledge cutoff.** If your web data is 2023-vintage, your model's knowledge is 2023-vintage. Adding recent data late extends the cutoff; requires careful handling.
- **Mix interacts with context length.** Code is much longer per document than tweets; math has specific token distributions. Packing behavior varies by domain.
- **Over-represented categories.** Web's default mix includes things you don't care about (spam, low-quality listicles, product reviews). Classification + downsampling is essential.
- **Benchmark contamination risk.** If your mix includes sources with benchmark test questions, you leak. Decontamination (see [decontamination](decontamination.md)) is a required step.
- **Mix isn't static across sizes.** 8B might benefit from 20% code; 405B might only need 17%. Mix scales with model size weakly — similar but not identical.
- **Labs don't publish detailed mixes.** Llama 3's ~50/25/17/8 is unusually detailed. OpenAI / Anthropic / Google generally don't publish this granularity.

---

## Sources

- Paper: *The Llama 3 Herd of Models* — Meta, 2024, arXiv 2407.21783, Sec. 3.1.2 — the 50/25/17/8 mix and scaling-law methodology.
- Paper: *Training Compute-Optimal Large Language Models (Chinchilla)* — Hoffmann et al., 2022 — compute-optimal at fixed mix; doesn't address mix itself.
- Paper: *Dolma: An Open Corpus of Three Trillion Tokens* — Soldaini et al., AI2, 2024, arXiv 2402.00159 — see [dolma](dolma.md).
- Paper: *Data Mixing Laws* — Ye et al., 2024, arXiv 2403.16952 — formal analysis of how mix proportions affect loss.
- Paper: *DoReMi* — Xie et al., 2023 — algorithm for learning an optimal data mix.
