# Loop-PP
*Depth — pipeline-parallel schedule for depth-recurrent / looped Transformers, where each stage is visited multiple times per micro-batch.*

**TL;DR:** Interleaved 1F1B is drawn for an acyclic stage graph in which each micro-batch enters each stage once. Depth-recurrent Transformers (see [loop-residuals](../architectures/loop-residuals.md)) visit the same stack several times per micro-batch, and later stages feed earlier ones — native 1F1B has no slot for the extra visit, cannot order a first iteration against a second, and its neighbor-only send/receive pattern has no back-edge to the first stage. Loop-PP (2026) rebuilds both timetable and communication for this cyclic layout, stays bit-identical to a single-stage reference, and on 8B–60B MoE models cuts peak memory by up to 32.8% while lifting throughput by 12.7%.

**Prereqs:** [README](README.md), [dualpipe](dualpipe.md)
**Related:** [../architectures/loop-residuals](../architectures/loop-residuals.md)

---

## What it is

A pipeline-parallel scheduler for models that iterate a shared block stack $L$ times per forward pass. The core modifications over interleaved 1F1B: distinguish iterations of the same stage as separate "virtual stages," order their execution consistently across ranks, and emit a static per-rank instruction stream that decouples send-post from receive-wait so communication overlaps compute.

## How it works

Three pieces:

1. **Virtual-stage expansion.** Each physical stage holds the same block weights; a looped stage is unrolled into $L$ virtual stages in the schedule. Each virtual stage has its own slot in the 1F1B skeleton, so there's no ambiguity about "which iteration of stage 3 fires now."
2. **Globally consistent order.** Loop-PP specializes 1F1B's F/B sequence into an order over virtual-stage visits that is consistent across ranks — critical because a later iteration on rank $r$ may need to receive from an earlier iteration on rank $r-1$, and a wrong order deadlocks.
3. **Async send/receive stream.** The schedule emits a static per-rank instruction stream where `send-post` is issued as early as possible and `recv-wait` as late as possible, maximizing overlap. All communication reaches back to the first stage via explicit back-edges because the loop requires stage-last-to-stage-first message passing.

The whole schedule is bit-identical to a single-stage reference, so correctness is not sacrificed for throughput.

## Why it matters

Without a cyclic PP schedule, every depth-recurrent architecture hits a wall at the first pipeline-partition point — a strong argument not to build them at all. Loop-PP removes that wall:

- Peak memory down by up to **32.8%** on 8B–60B MoE models.
- Throughput up by **12.7%** at matched compute budgets.
- Bit-identical to the non-pipelined reference.

Pairs naturally with [loop-residuals](../architectures/loop-residuals.md): LoopRes makes looping *useful*; Loop-PP makes it *trainable at scale*.

## Gotchas & tricks

- **Virtual-stage count explodes with $L$.** The 1F1B skeleton's warm-up and cool-down phases grow with virtual-stage count; for very deep loops, bubble costs can swallow gains.
- **MoE expert-parallelism composes but requires care.** Each virtual-stage visit still routes through expert-parallel all-to-all; scheduling has to avoid collisions across loops.
- **Interaction with recomputation.** Depth-recurrent models with activation checkpointing rematerialize per loop; naively this re-pays recomputation cost $L$ times. Loop-PP's schedule reuses rematerialized activations across loop visits where possible.
- **Static instruction streams** mean the schedule is fixed per model shape — reshaping (different $L$, different MoE routing) needs re-planning.

## Sources

- Paper: A Pipeline-Parallel System for Depth-Recurrent Models — Tencent, 2026 — [arXiv:2611.30581](https://arxiv.org/abs/2611.30581)
