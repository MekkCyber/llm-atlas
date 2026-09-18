# MoE Memory Peaks — Four Fixes for Long-Context MoE Training
*Depth — an enumeration of the four unbounded memory peaks in MoE training and exact-gradient fixes for each.*

**TL;DR:** Long-context MoE training doesn't fail on *average* memory — it fails on **whichever peak spikes first**. Four peaks are left unbounded by the parallelism plans in common use: **expert dispatch** (grows with routing matrix), **vocabulary projection** (tokens × vocab), **checkpoint boundaries** (depth × seq), **optimizer state** (parameter count). Pandit et al. (2026) introduce four surgical fixes — PipelinedLLEP, Ring-DTP, Selective Checkpoint Offload, OffloadStreamAdamW — that bound the GPU working set at launch and preserve exact loss and gradients. Together they train 120B–667B MoEs at **1M-token context**, 8–32× the reach of a tuned FSDP2 baseline.

**Prereqs:** [../architectures/_moe.md](../architectures/_moe.md), [../architectures/capacity-factor.md](../architectures/capacity-factor.md)
**Related:** [dualpipe](dualpipe.md) · [../fundamentals/z-loss.md](../fundamentals/z-loss.md) · [../pre-training/fp8-training.md](../pre-training/fp8-training.md)

---

## What it is

A four-part memory optimization for long-context MoE training. Each fix targets one of the four unbounded peaks; all four preserve mathematical equivalence with the baseline — only the *order and granularity* of computation and data movement change.

## How it works

### 1. PipelinedLLEP — expert dispatch peak

Extends least-loaded expert parallelism with a cap on the tokens each source contributes to a dispatch chunk. Instead of one all-to-all with the full routing matrix, dispatch in chunks whose size is fixed at launch. Peak allocation for the dispatch buffer becomes independent of batch × seq.

### 2. Ring-DTP — vocabulary projection peak

The logits matrix $B \cdot T \cdot V$ dominates memory at long context / large vocab. Ring-DTP shards logits around a ring and folds each shard into an **online log-sum-exp** (analogous to Ring Attention for softmax). The cross-entropy loss is computed streamingly; the full logits tensor is never materialized on any device.

### 3. Selective Checkpoint Offload (SCO) — activation-checkpoint boundary peak

Standard activation checkpointing recomputes activations on backward. Between forward and backward, the *one long-lived tensor per checkpoint boundary* still lives in HBM. SCO copies that single tensor to CPU memory (via async D2H) as soon as it's produced and prefetches it back at the exact moment backward needs it.

### 4. OffloadStreamAdamW — optimizer state peak

Standard "CPU AdamW offload" serializes the CPU Adam step across all parameters — bounded memory, but the CPU step becomes the wall-clock bottleneck. OffloadStreamAdamW turns the serial CPU update into a **bucket pipeline**: overlap CPU Adam on bucket $i$ with H2D copy of bucket $i-1$'s updated weights.

## Why it matters

- **Every peak, not the average.** Prior work reduced one peak at a time; whichever remained next-largest still OOM'd. Bounding all four unlocks context lengths that were previously impossible even in principle.
- **Component wins.** Dispatch peak −59.3% (no throughput loss), vocab projection peak −86.6%, offloaded optimizer step **2.05× faster**.
- **Composed wins on frontier scale.** Trains 120B–667B MoEs at **1M-token context**, 8–32× a tuned FSDP2 baseline, up to **10.4×** its throughput.
- **Exact gradients.** No approximation. Loss curves are identical to the (would-OOM) baseline.

## Gotchas & tricks

- **Which peak spikes first depends on your config.** Model size, context length, expert count, device count — the paper's own table shows the winning peak swaps across settings. Profile before picking which fix to prioritize.
- **Ring-DTP requires an online-safe loss.** Cross-entropy composes with log-sum-exp; auxiliary losses that use the full logits tensor (some z-loss formulations) need adaptation.
- **SCO's async copy needs headroom.** PCIe bandwidth must exceed the checkpoint-boundary tensor size / step time. On slow interconnects, prefetch latency dominates and SCO becomes a loss.
- **Bucket size for OffloadStreamAdamW is a hyperparameter.** Too small → per-bucket launch overhead dominates. Too large → back to serial. Paper reports empirically-tuned defaults.

## Sources

- Paper: *Flattening Every Memory Peak in Long-Context Mixture-of-Experts Training* — Pandit, Nguyen, Zhao, Joty, Salesforce Research, 2026, arXiv 2609.14306.
