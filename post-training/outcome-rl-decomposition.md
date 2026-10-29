# Outcome-RL Decomposition (Finishing / Reliability / Reach)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A measurement framework that explains **what pass@1 gains from outcome-reward RL actually come from**. Decomposes the pass@1 improvement over the base model into three disjoint parts: **finishing** (answering inside the token budget), **reliability** (being right once finished), and **reach** (solving problems the base model never solves). Applied to 20 released RL models (1.5B–32B) on physics and math under fixed token budgets, Yang (2026) reports that **reach adds at most 2.1 percentage points** across all cases, and most pass@1 gain is finishing or reliability. The implication: outcome RL rarely teaches models to solve problems they couldn't already touch.

**Prereqs:** [rlvr.md](rlvr.md), [_rl.md](_rl.md)
**Related:** [rlvr.md](rlvr.md) · [reasoning/long-cot-rl.md](reasoning/long-cot-rl.md) · [_rl.md](_rl.md)

---

## What it is

The headline evidence for outcome-reward RL on reasoning models is pass@1 improvements over the base model. But pass@1 can rise for different reasons, and they have very different implications for what RL "learned":

- **Finishing.** The base model was running out of tokens on hard problems; RL taught it to finish faster (shorter CoT, better pacing).
- **Reliability.** The base model finishes but sometimes answers wrong; RL made its answers on *already-solvable* problems more consistent.
- **Reach.** RL unlocked problems the base model fundamentally couldn't do.

The decomposition partitions each unit of pass@1 gain into exactly one bucket, so claims like "RL teaches reasoning" become testable: if gains are mostly finishing/reliability, the base already had the capability; if gains are reach, it didn't.

## How it works

For a fixed token budget $B$ and a benchmark of problems $\mathcal{P}$:

1. **Base model pass@1.** Run the base reasoning model on each problem. Classify its result as **finished-correct**, **finished-incorrect**, or **ran-out**.
2. **RL model pass@1.** Same under the RL-trained model.
3. **Per-problem transition table.** Each problem transitions between the two models through one of 9 cells (base state × RL state). The pass@1 improvement decomposes:

| Base \ RL           | finished-correct | finished-incorrect | ran-out |
| ------------------- | ---------------- | ------------------ | ------- |
| finished-correct    | 0                | *regression*       | *regression* |
| finished-incorrect  | **reliability**  | 0                  | 0       |
| ran-out             | **finishing** (if newly correct) · also **reach** if base can never solve | *finishing (incorrect)* | 0 |

**Reach** is the subset of "finishing-correct" problems where resampling or lookup confirms the base model *never* solved them under any budget — these are genuinely new capabilities, not just budget-management wins.

### Reach verification

Finishing-correct cases need an extra test to be classified as reach: resample the base model many times (or widen its budget), check whether it ever gets the right answer. If yes, it's finishing, not reach.

## Why it matters

- **Deflates "RL unlocks reasoning" claims.** Yang reports that at 4k–8k budgets, **82% of pass@1 gain comes from finishing**. At larger budgets, reliability takes over. Reach is almost always ≤ 2.1 pp.
- **Reach is small in a controlled study too.** In a dedicated 8B RL run reported in the paper, gains on new problems were pure finishing; reliability was flat. Rules out "didn't look hard enough."
- **Reframes the right metric.** If "did RL teach something new?" is the question, pass@1 is a poor proxy — most of its growth is budget management. **pass@$k$ at large $k$** is the right test for reach.
- **Policy implication.** Reasoning gains might come as cheaply from finishing-focused SFT (e.g. shorter-trace demonstrations) as from outcome RL. Worth measuring directly.

## Gotchas & tricks

- **Budget dependence is huge.** Finishing wins dominate under tight budgets. Loosening the budget shifts gain to reliability. Report the decomposition for *every* budget you cite.
- **Reach verification cost.** Confirming the base model "never" solves a problem is expensive — many resamples at wide budgets. Shortcuts (prompting tricks, external search) can misclassify reach as finishing.
- **Not a critique of RL itself.** Outcome RL is still useful if finishing and reliability are what you want. The paper's point is accuracy of description, not that outcome RL is bad.
- **Applies to *outcome* RL specifically.** Process-reward RL (per-step signals) may show different decompositions; the technique as published studies outcome-only recipes.
- **Base-model dependence.** Decomposition is relative to one specific base model; a different base can shift the bucket distribution without changing the RL training recipe.

## Sources

- Paper: *What Outcome RL Buys: Finishing, Reliability, Rarely Reach* — Yang, 2026 — [arXiv:2610.34894](https://arxiv.org/abs/2610.34894)
