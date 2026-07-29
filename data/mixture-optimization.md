# Data-Mixture Optimization (DecoupleMix)
*Depth — a two-stage optimization for training-data recipes: single-variable search over inter-class ratios, constrained convex allocation over intra-class datasets.*

**TL;DR:** Turn heuristic data-mixture construction into a reproducible optimization problem. Decompose the mixture into **inter-class ratios** across capability categories (line-search: one degree of freedom) and **intra-class ratios** within each category (constrained convex optimization scored by Quality × Difficulty with a diversity objective). Ratios optimized on small proxies transfer to larger scales without retuning; VLMs trained with **80B additional multimodal tokens** under DecoupleMix are competitive with strong open-source VLMs using substantially larger budgets.

**Prereqs:** [_data-curation.md](_data-curation.md), [quality-filtering.md](quality-filtering.md)
**Related:** [../multimodal/README.md](../multimodal/README.md)

---

## What it is

VLM (and LLM) data curation has largely been "stack whatever passes the quality filter and set ratios by intuition." That leaves no principled way to admit new data or validate claims about it. DecoupleMix treats mixture construction as a two-stage optimization that turns dataset validation into a controlled, attributable experiment.

## How it works

Two orthogonal sub-problems, each with the right tool:

1. **Inter-class ratios (capability categories).** A single-variable iterative search over the relative weight of each capability category (e.g. captioning vs OCR vs grounding vs chart understanding). Because it's one-dimensional per category, line search suffices; can be run on small proxy models and the discovered ratios transfer to larger models without retuning.
2. **Intra-class ratios (datasets within a category).** For each category, score every candidate dataset by **Quality** and **Difficulty**, then formulate selection as a *constrained convex optimization* with a diversity objective — pick the dataset weights that maximize weighted quality-difficulty subject to a diversity constraint.

The framework separates the two decisions cleanly: what capabilities to invest in (inter-class) vs which datasets serve each capability best (intra-class).

## Why it matters

- **Proxy-to-scale transfer** is the property that makes mixture optimization actually practical — you develop the recipe on cheap proxies and deploy at scale without another sweep.
- **Attributable validation.** Adding a candidate dataset becomes a controlled experiment inside the convex-selection stage rather than a heuristic call.
- **Empirical wins.** DecoupleMix beats heuristic baselines and, with only 80B additional multimodal tokens, matches strong open-source VLMs trained with substantially larger multimodal budgets.

## Gotchas & tricks

- Category taxonomy is the framework's implicit prior — if the taxonomy misses a real capability, no amount of optimization inside it will surface the gap.
- Quality × Difficulty scoring requires *per-dataset* scoring functions; these are non-trivial to build and are the pipeline's biggest engineering cost.
- Convex formulation admits many local knobs (diversity constraint form, difficulty scale); document them explicitly so the recipe is reproducible.
- Transfer only holds within the proxy-to-scale regime tested — very large scale-ups may still shift optimal ratios.

## Sources

- Paper: *DecoupleMix: Decoupled Ratio Search and Convex Allocation for Scalable VLM Data Recipes* — Xie et al., 2026 (ByteDance) — [arXiv:2607.24516](https://arxiv.org/abs/2607.24516)
