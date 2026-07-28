# DataPrep-Bench
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A benchmark that evaluates LLMs on *preparing training data* rather than answering downstream tasks. Two capability axes: (1) transforming raw sources into supervised training data, and (2) predicting a candidate dataset's downstream training value **before** actually running the training. Spans six domains with multiple base models. Ships two baselines — a skill-guided data-prep agent and a distributional quality metric.

**Prereqs:** [../data/_data-curation.md](../data/_data-curation.md), [../data/quality-filtering.md](../data/quality-filtering.md)
**Related:** [../data/decontamination.md](../data/decontamination.md), [../post-training/rejection-sampling.md](../post-training/rejection-sampling.md)

---

## What it is

An evaluation suite that measures how well an LLM can act as a *data preparator* for another LLM. Standard benchmarks score the model's own answers; DataPrep-Bench scores the training data the model produces and its judgments about candidate corpora. Six domains keep the axis broad enough that per-domain quirks don't dominate the score.

## How it works

- **Task 1 — Transformation.** Given raw sources (documents, transcripts, code), the model-under-test emits supervised training data. Downstream lift on a target task is the primary score.
- **Task 2 — Value prediction.** Given a candidate training dataset, predict how much a base model would improve after training on it — *without* training. Correlation with the actual post-training delta is the score.
- **Baselines.** A skill-guided *DataPrep Agent* iteratively decomposes preparation into sub-skills. A *distributional quality metric* scores candidate datasets by their feature-statistic distance from a strong reference distribution.

## Why it matters

Post-training pipelines lean heavily on LLM-generated synthetic data with almost no principled way to predict whether the new mixture will help. A working value predictor (Task 2) short-circuits an expensive iteration loop: filter mixtures before you burn a training run on them. The benchmark also produces one negative-headline result — domain-specific synthetic data *frequently underperforms* general corpora — that undermines a common assumption in vertical-domain fine-tuning.

## Gotchas & tricks

- **Value prediction is the harder axis.** Transformation quality is easier to game with prompt engineering; the predictor task requires the model to actually reason about training dynamics.
- **Distributional metric > LLM-as-judge.** In most of the six domains, the training-free distributional metric had *higher* cross-model correlation with real training outcomes than LLM-generated quality scores — a datapoint against LLM-as-judge quality scoring for training data.
- **Six domains only.** Coverage is broad but not exhaustive; a model tuned to the benchmark's domain mix may not transfer.

## Sources

- Paper: *DataPrep-Bench: Benchmarking LLMs as Training Data Preparators* — Cai, Lin, Du, Xia, Qiu, Sun, Qiang, Han, Ma, Zeng, An, He, Zhang (Peking University et al.), 2026 — [arXiv:2607.20465](https://arxiv.org/abs/2607.20465).
