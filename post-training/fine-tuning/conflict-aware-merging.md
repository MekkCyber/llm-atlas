# Conflict-Aware Merging (for Instruction Tuning)

*Depth — partition a heterogeneous instruction-tuning mixture along its top gradient-conflict axes, fine-tune partitions independently with no communication, merge once at the end.*

**TL;DR:** Joint SFT on heterogeneous instruction mixtures suffers from gradient interference (tasks pull weights in conflicting directions) and bandwidth-heavy synchronization. MERIT (NAVER AI 2026) jointly addresses both: estimate dataset-level gradient conflicts, partition the mixture along the top PCA axes of that conflict matrix, fine-tune each partition independently in parallel with zero inter-partition communication, then merge the resulting checkpoints once via token-weighted averaging. Backed by a local-quadratic theory of weight merging as curvature-weighted variance reduction.

**Prereqs:** [../../pre-training/model-souping.md](../../pre-training/model-souping.md), [../_post-training.md](../_post-training.md)
**Related:** [../_rl.md](../_rl.md), [../rejection-sampling.md](../rejection-sampling.md)

---

## What it is

The standard joint SFT loop pays two costs as the instruction mixture grows:

| Cost | Where it shows up | Why |
| --- | --- | --- |
| **Gradient interference** | Joint training plateaus; per-task accuracy underperforms per-task SFT | Conflicting gradients cancel each other |
| **Bandwidth** | Multi-node SFT throughput collapses on heterogeneous data | All-reduces dominate step time |

Conflict-aware merging proposes that both stem from the same underlying problem: the mixture has *structurally different* gradient directions. If you can identify those directions and partition the data so each partition pulls in a coherent direction, you can train each partition independently (no comms) and merge once (no interference) — and the theory says merging is in the same shared flat basin.

## How it works

Four steps:

1. **Estimate dataset-level gradient conflicts.** For each pair of sub-datasets / tasks, compute a conflict score from their gradients on a shared base model. Assemble into a conflict matrix.
2. **PCA-aligned splitting.** Take the top eigenvectors of the conflict matrix as conflict axes. Partition the mixture along these axes — examples projecting onto the same axis go to the same partition.
3. **Independent fine-tuning.** Each partition fine-tunes a copy of the base model with *no* gradient sync to the others. Embarrassingly parallel — partitions can live on different nodes / clusters / clouds.
4. **One-shot merge.** Token-weighted average of the partition checkpoints into a single model. No iterative refinement, no fine-tuning after merge.

The theory (local quadratic in a shared flat basin):
- Weight merging → curvature-weighted variance reduction. Averaging $K$ partition optima in a shared basin reduces variance proportional to local Hessian eigenvalues.
- PCA-aligned conflict splitting → maximizes that variance reduction along high-curvature directions, where it matters most.
- Merging → spectral filtering with implicit norm regularization.

## Why it matters

- **Bandwidth disappears.** Partitions never sync. Heterogeneous data centers / asymmetric clouds become usable for SFT.
- **Joint-training accuracy without joint-training compute.** On Qwen2.5-VL-3B with 136 Vision-FLAN tasks, MERIT lifts the 8-benchmark average from 54.3 (joint) to 57.0 — *better* than joint, with no inter-partition comms. Scales to a 7B model on 1.6M examples / 176 sources with minimal cost overhead vs joint.
- **Same recipe, vision and text.** Transfers from VLM SFT (Vision-FLAN) to text-only FLAN without recipe changes.
- **Theory grounds the practice.** The variance-reduction story makes the empirical "merging just works" mystery explicit — and identifies *when* it works (shared flat basin) and *when* it shouldn't (large basin distance).

## Gotchas & tricks

- **Conflict matrix is the bottleneck.** Estimating per-pair gradient conflicts on every pair of tasks is $O(K^2)$ in dataset chunks; in practice MERIT uses chunked estimates with random pairings.
- **Token-weighting beats uniform.** Naive uniform-weight averaging underweights large partitions; token-count weighting is the simplest fix and matches the theory.
- **Shared base is non-negotiable.** All partitions must start from the same checkpoint — otherwise the "shared basin" assumption fails and the merge degrades.
- **Number of partitions is a tuning knob.** Too few partitions → residual conflict within partitions. Too many → variance from small partitions dominates. The paper sweeps a small range; per-mixture tuning is recommended.

## Sources

- *Decentralized Instruction Tuning: Conflict-Aware Splitting and Weight Merging* — Choi & Kim, NAVER AI, 2026 — [arXiv:2606.01717](https://arxiv.org/abs/2606.01717) — primary source. MERIT recipe, local quadratic theory, Vision-FLAN + FLAN experiments.
- *Model Soups* — Wortsman et al., 2022 — original weight-averaging baseline. See [../../pre-training/model-souping.md](../../pre-training/model-souping.md).
- *TIES-Merging / DARE* — 2023–2024 — task-vector merging lineage.
