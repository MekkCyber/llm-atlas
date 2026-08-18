# Dion3
*Depth — full-stack overhead reduction for Muon-style orthogonal-update optimizers.*

**TL;DR:** Dion3 is a drop-in Muon replacement that attacks the orthogonalization overhead at every level of the stack: a Gram-form Newton-Schulz cuts FLOPs, symmetry-aware CuteDSL kernels cut kernel time, a megabatching strategy cuts sharded-communication overhead, and a **row-subset** update rule orthogonalizes only a fraction of the momentum matrix's rows per step. Matches or improves Muon's training loss with up to **6× faster** optimizer steps.

**Prereqs:** [_training-stability](_training-stability.md)
**Related:** [fp8-training](fp8-training.md), [../systems/dualpipe.md](../systems/dualpipe.md)

---

## What it is

The Muon optimizer improves loss over AdamW on transformer pretraining by orthogonalizing (via Newton-Schulz) the momentum matrix before applying the update. The cost: cubic-time NS iterations, expensive at scale, and — when weights are sharded — the orthogonalized matrix has to be reassembled across ranks, which requires all-gather traffic that erodes the speed-of-loss advantage. Dion3 keeps Muon's convergence properties while cutting all three overheads.

## How it works

Four independent optimizations, each safe in isolation, compounding when stacked:

1. **Gram Newton-Schulz.** Instead of iterating on the full `d × d` momentum matrix `M`, iterate on `G = M Mᵀ` (`d × d` but symmetric) or its smaller counterpart. Recovers `M`'s orthogonal factor from `G`'s square-root, saving FLOPs.
2. **CuteDSL kernels.** Custom CUDA kernels written in CuteDSL exploit the symmetry of `G` and of intermediate NS iterates, roughly halving kernel time vs a generic matmul.
3. **Megabatching.** Batch several optimizer steps' worth of gradient statistics before triggering the sharded-comm phase — amortizes all-gather cost across steps.
4. **Row-subset orthogonalization.** Each step, sample a fraction `r < 1` of `M`'s rows and orthogonalize only those; leave the rest untouched. Empirically this preserves Muon's loss trajectory while cutting NS work proportionally.

Ships as the `dion` package with a `Dion3` optimizer class that matches Muon's API.

## Why it matters

- **Removes the last argument against Muon at scale.** Muon-family already beats AdamW on loss for transformer pretraining; the overhead was the ship-blocker for frontier labs. 6× faster steps and same loss means Muon becomes the default at scale.
- **Composes with FP8 and sharded training.** Because the wins are at the optimizer-step layer (not the forward/backward), Dion3 stacks with FP8 pretraining and standard FSDP/TP.
- **Row-subset is a general lever.** The idea "orthogonalize a subset per step" applies to any orthogonal-update method, not just Muon.

## Gotchas & tricks

- Row-subset rate `r` is the main tuning knob — too aggressive and loss regresses to Muon minus subsampling; too conservative and you're back at Muon's cost.
- Megabatching interacts with checkpoint frequency: batching optimizer steps means a mid-batch restart replays more work.
- Gram NS numerics need care in FP8 / BF16; the paper's kernels handle this but naive reimplementations can diverge.

## Sources

- Dion3: Full-Stack Orthogonal Updates — Noah Amsel, Jack Zhang, Kwangjun Ahn, Ali Naeimi, Austin Feng, Berlin Chen, Tri Dao, John Langford — 2026 — [arXiv:2608.11612](https://arxiv.org/abs/2608.11612)
- Code: `dion` package (drop-in Muon replacement).
