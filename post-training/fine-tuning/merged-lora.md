# Merged LoRA (for continual fine-tuning)
*Depth — using per-task LoRA adapters merged into shared weights as a continual-learning allocation rule.*

**TL;DR:** Merged LoRA is a continual-fine-tuning **allocation** in which each new task gets its own LoRA adapter that is then merged (weight-averaged or task-averaged) into a shared pool, rather than trained monolithically or stacked. Composed with data / function / weight anchors, this is the recipe Zhang et al. (2026) find raises 100-task sequential-SFT retention from 1.2% to 34.9% — a 28× improvement over naive sequential fine-tuning, with no replay of prior tasks.

**Prereqs:** [../../pre-training/model-souping.md](../../pre-training/model-souping.md), [README](README.md)
**Related:** [../_post-training.md](../_post-training.md)

---

## What it is

A recipe for **long-horizon SFT** — training a single base model on a long sequence of tasks without retaining prior task data. Merged LoRA sits at the intersection of two ideas: **LoRA** (train a low-rank additive update per task) and **model souping** (average multiple trained models to get a stronger one). Instead of one big adapter that gets clobbered by the newest task, each task's LoRA is folded into a running average.

## How it works

For each task `t` in a stream of `T` tasks (no data replay):

1. **Start from the current base weights.** The base is the pretrained model plus any previously merged adapters folded into the weights (`W_base ← W_pretrained + Σ_{i<t} α_i · A_i B_i`).
2. **Train a LoRA adapter** `A_t B_t` on task `t` with the standard low-rank objective.
3. **Merge.** Fold the new adapter into the base weights with a small mixing coefficient: `W_base ← W_base + α_t · A_t B_t`. `α_t` is typically small (e.g. 1/N-ish) to prevent the new task from overwriting older ones.
4. Composed with **three anchors**:
   - **Data anchor.** A small buffer of proxies (synthetic representatives, not raw replay) keeps the loss landscape near prior tasks.
   - **Function anchor.** Regularizer on output distributions of prior tasks.
   - **Weight anchor.** Elastic constraint keeping updated weights near the base.

Together, `(anchors × merged LoRA)` outperforms any single trick.

## Why it matters

Long-horizon SFT is the current bottleneck for "always-learning" assistants: you want to fine-tune your production model on Monday's data on Monday, on Tuesday's data on Tuesday, and still remember January. Replay-based methods store data; merged LoRA doesn't. On the Zhang et al. 100-task benchmark:
- Naive sequential FT: **1.2%** final retention.
- Best merged-LoRA + three-anchor composition: **34.9%** — top-3 across all evaluated datasets.

## Gotchas & tricks

- **Merging coefficient `α_t` matters.** Too high → catastrophic forgetting; too low → new task not learned. The paper uses a schedule proportional to task loss slope.
- **Rank of LoRA per task.** Small ranks (r=8–16) compose better than large ones.
- **Merged LoRA ≠ model souping.** Model souping averages *fully trained* models; merged LoRA averages the *adapters*. Both share the intuition that averaging in the linear regime lands in a wider basin.
- **Compositions matter more than any anchor alone.** The paper's central finding is that the three anchors combined with merged LoRA compound — no single anchor + merged LoRA reproduces the 34.9%.
- **Order of tasks matters.** A hard-first schedule leaves more capacity for later merges than easy-first.

## Sources

- Paper: *Continual Learning Mechanisms Compose for Long-Horizon Memorization* — Zhang, Khashabi, Shu — 2026 — [arXiv:2609.06986](https://arxiv.org/abs/2609.06986).
- Related paradigm: model souping — Wortsman et al., 2022.
