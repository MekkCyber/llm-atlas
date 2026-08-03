# MoE Routing Analysis
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** The common story — "MoE routing works because co-selected experts occupy geometrically **disjoint** subspaces" — turns out to conflate three separate quantities: **route coherence**, **candidate quality**, and **candidate–context interaction**. Tian, Xu, Li (ZJU / PolyU, 2026) introduce a measurement stack (**ESSI**, matched-route residuals, prefix-controlled 2×2 factorial) and find **coherent overlap** — expert subspaces overlap heavily, yet multi-expert composition still improves next-token prediction. Geometric redundancy ≠ functional redundancy.

**Prereqs:** [_moe](_moe.md), [deepseek-moe](deepseek-moe.md), [load-balancing-loss](load-balancing-loss.md)
**Related:** [aux-loss-free-balancing](aux-loss-free-balancing.md), [capacity-factor](capacity-factor.md)

---

## What it is

MoE routing analysis asks: *why does Top-k routing beat Top-1?* The default answer — "each expert contributes a distinct direction" — assumes geometric complementarity between co-selected experts. But three effects have been conflated:

- **Route coherence** — do the router's chosen experts actually cover different subspaces?
- **Candidate quality** — even at fixed routes, would the model do better if a different expert had been picked?
- **Candidate × context interaction** — does prefix context change which experts are useful?

Untangling these matters for pruning proposals (if experts are geometrically redundant, some can be dropped) and for interpretability (do experts *specialize* or just *cover*?).

## How it works

**Expert Subspace Separation Index (ESSI).** For each token, measure the geometric overlap between co-selected experts' output subspaces. Low ESSI = redundant; high ESSI = complementary.

**Matched-route residuals.** For each actual routing decision, compute what the residual representation *would* be if a matched alternative expert were selected instead. Measures counterfactual quality of unselected candidates.

**Prefix-controlled 2×2 factorial.** For each of the actual/alternative expert choices, condition on either the actual prefix or a control prefix, giving 4 cells. Isolates the interaction between routing choice and context.

**Frozen-route Top-k ablation.** Freeze the router at Top-1; incrementally add the Top-2, Top-3, ..., experts one at a time and re-measure next-token prediction. If adding a later expert improves prediction, it's *functionally* useful even if geometrically overlapping.

Applied across **OLMoE, Mixtral, DeepSeek** MoE architectures.

## Why it matters

- **Overturns a load-bearing intuition.** Papers that treat geometric complementarity as the mechanism (and use it to justify pruning or router changes) are targeting the wrong quantity.
- **Adding Top-2 improves prediction in 24/39 frozen-route cases** — functional benefit persists without disjoint linear coverage.
- **Concrete measurement stack.** ESSI + matched-route residuals + factorial is reusable for future MoE analysis and pruning-value estimation.
- Implies **conservative pruning:** don't cut experts because their outputs overlap geometrically — measure functional contribution frozen-route first.

## Gotchas & tricks

- ESSI depends on the subspace definition — use SVD of expert output samples, not weight-space distance.
- Matched-route counterfactuals require the router logits, not just the argmax — implementations that only expose Top-k mask lose the alternatives.
- Prefix-control experiments need a *neutral* control prefix; a well-chosen but non-informative prefix is not the same as a length-matched random prefix.
- Frozen-route ablation is compute-heavy at scale — sample tokens strategically.

## Sources

- Paper: *Beyond Geometric Complementarity: Coherent Overlap in Sparse Mixture-of-Experts Routing* — Tian, Xu, Li, ZJU / HK PolyU, 2026 — [arXiv:2607.28308](https://arxiv.org/abs/2607.28308).
