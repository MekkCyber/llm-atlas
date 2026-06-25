# Agentic Data Curation
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A multi-stage pipeline for assembling training data that produces *broadly capable* agentic models, not specialists tuned to a single benchmark. OpenThoughts-Agent (Raoof et al., 2026) runs 100+ controlled ablations across task sourcing, diversity expansion, trajectory generation, and filtering; the resulting 100K-example set lifts Qwen3-32B to 44.8% mean accuracy across seven agentic benchmarks (vs. 40.9% for the prior best open-data agent, Nemotron-Terminal-32B).

**Prereqs:** [_data-curation.md](_data-curation.md), [quality-filtering.md](quality-filtering.md)
**Related:** [../agents/README.md](../agents/README.md) · [../post-training/_post-training.md](../post-training/_post-training.md) · [../post-training/rejection-sampling.md](../post-training/rejection-sampling.md)

---

## What it is

Existing open agentic-data efforts (SWE-Smith, SERA, Nemotron-Terminal) optimize for one benchmark and produce specialists. OpenThoughts-Agent targets *generalization* across diverse agentic tasks — SWE-bench, TerminalBench, OSWorld, tool-use, browsing — by treating data curation as a pipeline whose stages can each be ablated.

## How it works

Four stages, each with its own ablation axis:

| Stage | Decision | Ablation finding (paraphrased) |
| --- | --- | --- |
| Task sourcing | Which benchmarks / synthetic generators feed the pipeline | Source diversity beats source quality past a threshold; narrow sources produce specialists |
| Diversity expansion | Mutate tasks across difficulty, domain, tool set | Type-aware mutation outperforms generic paraphrasing |
| Trajectory generation | Which model generates the agent traces; how many rollouts per task | Rollouts from a stronger model dominate; multi-rollout filtering helps |
| Filtering | Quality, correctness, novelty filters | A correctness filter is necessary; novelty filters help only after diversity is already high |

The full pipeline is run open-source — every ablation experiment, the resulting 100K-example mix, and the fine-tuned Qwen3-32B checkpoint are released. Data curation as a *reproducible scientific object*, not just an artifact.

## Why it matters

- The agentic equivalent of OpenThoughts/Tülu — an open, ablation-justified data recipe that gives the rest of the community a defensible starting point.
- Quantifies a lesson the field has been operating on by intuition: source *diversity* dominates source *quality* once basic correctness filters are in place.
- The 32B-fine-tune tier is now competitive with closed proprietary stacks on agentic benchmarks without any proprietary data.

## Gotchas & tricks

- The 44.8% headline is across seven specific benchmarks; weight the result by your downstream task mix.
- The pipeline assumes a strong rollout-generation model (a frontier agent). Cheaper teachers degrade the dataset.
- Pure SFT on this data; pairing with agentic RL is left as future work.

## Sources

- Paper: *OpenThoughts-Agent: Data Recipes for Agentic Models* — Raoof, Zhuang, Nezhurina, Guha, et al., 2026 — [arXiv:2606.24855](https://arxiv.org/abs/2606.24855).
- Project: openthoughts.ai (data, pipeline, ablations, models).
