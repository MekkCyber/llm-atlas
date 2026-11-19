# Owner-Based Muon
*Depth — sharded Muon that assigns one rank per matrix to run Newton–Schulz, instead of duplicating it everywhere.*

**TL;DR:** Muon's Newton–Schulz orthogonalization step needs the *full* matrix, which in standard sharded setups is duplicated across every rank. Owner-based Muon picks **one owner rank per matrix** that assembles the full matrix via P2P, runs NS, and scatters the result back. Parameters and momentum stay sharded; only the NS compute moves. Cuts step time by ~10% at frontier scale with no quality risk.

**Prereqs:** [hsdp](hsdp.md)
**Related:** [fp8-training](fp8-training.md), [_training-stability](_training-stability.md)

---

## What it is

A **sharded optimizer implementation** for Muon. Muon's update rule includes a Newton–Schulz iteration that approximates the matrix sign / orthogonalization of the momentum tensor:

```
# Muon core step (simplified)
M ← momentum tensor for this parameter matrix
U ← NewtonSchulz(M, k)      # k iterations: U ← (3/2)U − (1/2) U U^T U
W ← W − η · scale · U
```

The NS step is a **full-matrix** operation — it needs the whole `M`, not a shard. The obvious distributed implementation has every rank AllGather the shards, run NS locally, and keep only its own slice of the result. Correct but wasteful: every rank does the full NS, redundantly.

## How it works

Assign each matrix one **owner rank** within its shard group. Non-owners **P2P-send** their shards to the owner. The owner assembles the full input, runs NS once, applies the Muon scale, keeps its own slice, and **scatters** the rest back to the other ranks.

```
owner rank o of matrix W (parameters sharded across ranks r_1..r_n):
    for r in {r_1..r_n} \ {o}: receive shard_r from r
    M_full  = assemble({shard_r})
    U_full  = NewtonSchulz(M_full, k)
    for r in {r_1..r_n} \ {o}: send slice_r(U_full) to r
    keep slice_o(U_full), apply update

non-owner rank r:
    send shard_r to owner
    receive slice_r from owner
    apply update
```

**Ownership** is assigned per matrix, and replica-local groups can choose different owners. Matrices are **grouped by shape, shard group, and estimated cost**; compatible-shape matrices are stacked for batched matmul inside the owner, though each is still orthogonalized independently.

**State management.** Parameters, momentum, and master weights stay in their original sharded layout. Only the full-matrix computation (NS) moves to the owner. No extra persistent memory for the owner — buffers are per-phase with explicit memory budgets.

**Scheduling.** Comm is batched — several P2P sends fuse where possible. Waits happen only where the owner needs data. Owners are staggered across matrices so no single rank is swamped.

## Why it matters

Measured on the Xing 4 285B / 1024-NPU profiler, with HSDP topology held fixed:

| Configuration | Step time |
|---|---|
| HSDP baseline (duplicated Muon) | 31.8 s |
| HSDP + owner-based Muon | **28.9 s** (−9.1%, 1.10×) |

9% step-time reduction at frontier scale, from a change that touches only the optimizer implementation. No known quality cost: the NS result is the same (modulo floating-point non-associativity from batching).

More importantly, it's a **transferable technique**. Any shard-aware optimizer that relies on a full-matrix step (Muon, future second-order methods like Shampoo / SOAP) can apply the same owner-based pattern. The naive implementation duplicating the full-matrix step is a surprisingly common waste that has gone uncorrected at frontier scale.

## Gotchas & tricks

- **Owner concentration.** Compute concentrates on owners during NS. For heterogeneous matrix shapes the slowest owner dominates step time; load-balance owners across ranks or small matrices will straggle.
- **Batching changes bitwise results.** Stacking compatible shapes for batched matmul inside the owner alters floating-point accumulation order. The paper notes NS results differ bitwise from the duplicated version; empirically no quality effect, but reproducibility runs need to disable batching.
- **Interplay with checkpointing.** State is still sharded, so checkpoints look identical to standard sharded Muon. Resume from a standard-Muon checkpoint works; the only difference is how the step is executed.
- **Doesn't help for small models.** At small matrix sizes the NS step is cheap; the P2P round-trip becomes net negative. Worth it only at `d_model × expert_dim ≥ 4096 × 2048`-class matrices, which is the frontier-MoE regime.
- **Requires P2P in the comm library.** Not every shard-aware optimizer framework exposes directed P2P — some rely on collectives only. On frameworks without P2P you get the degraded form: owner-only compute with full AllGather, which recovers most of the win.

## Sources

- Paper: *TeleChat to Xing4: Advancing Frontier Agentic Models on Domestic Chips and Frameworks* — TeleAI, 2026 — [arXiv:2611.22115](https://arxiv.org/abs/2611.22115)
- Primary Muon reference: Keller Jordan et al. (2024).
