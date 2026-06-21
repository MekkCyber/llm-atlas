# Predictive Validity
*Depth — ranking agent / LLM configurations by their generalization to unseen task batches, not by in-sample mean.*

**TL;DR:** Static leaderboards rank configurations by in-sample mean and assume the winner generalizes. **Predictive validity** measures generalization directly as `corr(rank_in_sample, rank_out_of_sample)` across configuration variants. In Patel et al.'s 60-author MCP-agent study (arXiv 2606.19704), in-sample mean and predictive validity were nearly uncorrelated — i.e., the leaderboard winner is often the wrong deployment choice. A methodological correction to how the field reports agent and LLM evaluations.

**Prereqs:** [README](README.md)
**Related:** [livecodebench](livecodebench.md) · [../agents/README.md](../agents/README.md)

---

## What it is

A statistic for evaluating *configurations* (a model + orchestration + retrieval + reasoning + infra bundle) rather than individual completions. The score answers a simple question: if I pick the configuration that looks best in this benchmark, will it also be best on a new batch of tasks I haven't seen?

Defined as:

```
predictive_validity(config) =
    corr( rank(config, eval_batch_A) , rank(config, eval_batch_B) )
```

where `A` and `B` are disjoint samples from the same task distribution. A configuration with high in-sample mean but low predictive validity is overfitted to the specific eval split.

## How it works

In the source paper, 14 parallel implementation studies vary asset class, orchestration mode, retrieval strategy, reasoning mode, and infrastructure on one MCP-based industrial-agent benchmark. For each configuration, compute:

1. Mean score on a held-out in-sample batch (the leaderboard view).
2. Mean rank stability across batched resamples (predictive validity).
3. Cross-batch rank correlation against a new task distribution (out-of-sample predictive validity).

Configurations are then re-ranked by (2) or (3). The paper's headline: a configuration with the top in-sample mean often sits in the middle of the predictive-validity ranking. Selecting by predictive validity transfers better to the multi-modal visual extension they introduce.

## Why it matters

- **Leaderboards lie about deployment.** "Best on benchmark" rarely equals "best in production" when configurations are tuned per benchmark.
- **A measurable correction.** Predictive validity is a single scalar reviewers can demand, just like aggregate accuracy.
- **Generalizes across eval families.** Any benchmark with enough tasks for batched resampling can be re-scored this way — code, math, multi-turn agents, multimodal.

## Gotchas & tricks

- **Needs enough tasks per batch.** Predictive validity is a statistic over rankings; tiny benchmarks (≤30 tasks) yield noisy correlations.
- **Doesn't replace task-level error analysis.** Knowing "this config generalizes" doesn't tell you *why* the leaderboard winner didn't.
- **Sensitive to the configuration space.** The metric only measures generalization across the configurations you actually evaluated; can't extrapolate outside the swept space.
- **Aggregate-mean is still useful** for the chosen configuration — predictive validity is for *selection among* configurations.

## Sources

- Paper: *Beyond Static Leaderboards: Predictive Validity for the Evaluation of LLM Agents* — Patel, El Maghraoui, Lin, Li, Feng, Tsai, Sun, Xin et al. (60+ author consortium), 2026, arXiv 2606.19704.
