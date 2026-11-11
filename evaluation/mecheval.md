# MechEval (world-model exposures)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A per-mechanic, per-exposure scoring protocol for world models. On ARC-AGI-3, labels **69 recurring game mechanics across 20 games** from the underlying programs; for every model prediction, scores only the cells a mechanic actually changes, and tracks how many exposures the model needed to learn each one. Paired with WARC (a 35M-parameter ViT world model with test-time parameter updates), it shows test-time training triples an explorer's ARC-AGI-3 efficiency.

**Prereqs:** [./README.md](./README.md)
**Related:** [../pre-training/mid-training.md](../pre-training/mid-training.md), [../multimodal/README.md](../multimodal/README.md)

---

## What it is

World-model evals usually score full-frame prediction accuracy, which conflates "the model didn't move the wall" (easy) with "the model correctly predicted the new mechanic" (hard). MechEval disentangles the two by labeling, directly from the game's source, which cells each mechanic changes — and scoring *only those cells*. Each occurrence of the mechanic in gameplay is one **exposure**; the protocol asks how many exposures the model needs to reach $\ge 80\%$ cell-level accuracy on that mechanic.

## How it works

1. **Mechanic labeling (offline).** For each of 20 ARC-AGI-3 games, read the game's program and enumerate the recurring rules (how an action changes which cells). Canonical labels give 69 mechanics across the 20-game set.
2. **Per-prediction scoring.** On each gameplay transition, the model predicts the next frame. MechEval looks up which cells the active mechanic changes and scores the prediction only on those cells.
3. **Per-exposure tracking.** For each mechanic, count the number of gameplay occurrences the model has seen so far. Plot accuracy-vs-exposures. Report per-mechanic: (a) accuracy at the end of a play, (b) exposures to reach 80 %, (c) whether reached at all.
4. **Model side: WARC.** A 35M-param ViT pretrained on other games that updates **all** parameters after every observed transition at test time — test-time training as the generalization lever, not pretraining scale.

## Why it matters

- **Quantifies "test-time training" gain.** On ARC-AGI-3, four in five mechanics reach ≥80 % cell accuracy with test-time updates; without them, only one in seven does. Half reach the bar within ~80 exposures.
- **Explorer boost.** WARC-Agent (WARC + fixed graph explorer) triples the explorer's ARC-AGI-3 efficiency score (0.15 → 0.46 on public games; 0.20 on Kaggle hidden set).
- **Scales the "world model" debate.** Separates "the model has a pretrained prior" from "the model can learn a new mechanic in a few exposures", which matters for every claim about agents in novel environments.

## Gotchas & tricks

- **Mechanic labels are hand-written from game source.** Transferring MechEval to another benchmark requires equivalent per-mechanic labeling, which is work.
- **Cell-level accuracy ignores value** — a model that predicts "something changes here" but with the wrong new value gets zero credit; score at the right granularity for your claim.
- **Test-time training destroys the usual notion of a frozen checkpoint.** WARC's "model" is really "model + optimizer state + exposure history". Reproducibility requires logging the full trajectory.
- **Sample efficiency per mechanic is highly skewed.** Reporting only "fraction of mechanics reached" hides a long tail; the paper reports per-mechanic exposure curves.
- **Adjacent but distinct from benchmarking generation quality** — MechEval is a world-model prediction benchmark, not a video-quality benchmark; it complements rather than replaces [../multimodal/README.md](../multimodal/README.md)-style perceptual evals.

## Sources

- Paper: *How Many Exposures Does a World Model Need? Learning Unseen Game Mechanics at Test Time* — Kawano, Takashiro, Hayashi (Third Intelligence / U. Tokyo), 2026 — [arXiv:2611.12324](https://arxiv.org/abs/2611.12324)
