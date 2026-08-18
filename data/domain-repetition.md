# Domain Data Repetition (Scaling Law)
*Depth — how many epochs to repeat scarce high-quality domain data as model size grows at fixed tokens-per-parameter.*

**TL;DR:** When model size and total training tokens scale together at fixed tokens-per-parameter (TPP), the *optimal number of times to repeat a scarce domain corpus* mildly **increases** with model size — the opposite of the naive "bigger models overfit faster" intuition. Across domains, the sweet-spot repetition count is strongly negatively correlated with the domain's final validation loss (easier domains tolerate more repetition), and only weakly correlated with how much unique domain data you have.

**Prereqs:** [_data-curation](_data-curation.md), [quality-filtering](quality-filtering.md)
**Related:** [deduplication](deduplication.md), [dolma](dolma.md)

---

## What it is

A scaling-law study of the data-repetition question in the practical regime where the training-token budget grows proportionally with model size. The training mixture is *general web + a scarce high-quality domain corpus*; the question is how many times to cycle through the domain corpus as the total budget grows. Prior "repeat is bad past N epochs" results came from fixed-budget experiments — this paper reruns them under fixed-TPP scaling.

## How it works

Empirical protocol:

- Fix TPP (e.g. 20 or 40 tokens per parameter, chinchilla-adjacent).
- Vary model size across a small grid (proxy models: 100M–1B).
- For each model size, sweep the domain-repetition count `r` while holding the general-web mixture ratio fixed.
- Measure final validation loss per (model size, `r`) cell.
- Read out `r*(model size)` — the optimal repetition count as a function of scale.
- Repeat across several domain corpora with different intrinsic difficulty.

Two findings drop out:

1. **Fixed-domain sweep.** `r*` mildly *increases* with model size. Bigger models can absorb more passes over the same scarce corpus.
2. **Cross-domain sweep.** `r*` is strongly negatively correlated with the domain's final validation loss — "easy" domains (low achievable loss) tolerate more repetition than "hard" ones.

Practical implication: tune `r` on a small proxy model at the target TPP; that `r*` transfers to the larger model with the same TPP.

## Why it matters

- **Frontier data recipes.** Every big pretraining run has a scarce high-quality slice (curated code, math, papers, curated web) whose ratio dilutes as total tokens grow. Naive fixes are (a) filter more web (expensive), (b) generate synthetic (risky), or (c) repeat. This paper says (c) scales better than intuition suggests.
- **Cheap calibration.** Small-proxy tuning gives a defensible `r` estimate for the large run without expensive ablations at target scale.
- **Overturns a common heuristic.** "Don't repeat more than 4 epochs" comes from fixed-budget experiments; at fixed TPP with growing model size, the safe repetition count grows.

## Gotchas & tricks

- The law is stated *at fixed TPP*. Off-TPP training (e.g., undertrained large models) has a different repetition-scaling shape.
- "Domain validation loss" as the predictor of `r*` requires actually measuring it — a small proxy training run per domain.
- Interaction with data deduplication is subtle: repeating a corpus that hasn't been internally deduped inflates duplicates further; the paper's scaling law assumes upstream dedup.
- Very hard domains (high floor loss) can still overfit under moderate repetition; the negative correlation is a heuristic, not a guarantee.

## Sources

- Scaling Domain Data Repetition in LLM Pretraining — Jingwei Li et al., 2026 — [arXiv:2608.14071](https://arxiv.org/abs/2608.14071)
