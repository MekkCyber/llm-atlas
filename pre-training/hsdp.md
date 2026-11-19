# HSDP (Hybrid Sharded Data Parallelism)
*Depth — replace pipeline parallelism with FSDP-within-SuperPoD + replicate-across-SuperPoDs.*

**TL;DR:** HSDP shards model parameters with **FSDP within each SuperPoD** and **replicates across SuperPoDs**. Per-layer collectives stay on the fast intra-SuperPoD fabric; only coarse gradient reduction crosses SuperPoDs. With zero-copy per-parameter collectives and heavy backward overlap, it matches or beats pipeline parallelism at frontier scale without the pipeline bubble.

**Prereqs:** [../systems/dualpipe](../systems/dualpipe.md)
**Related:** [_training-stability](_training-stability.md)

---

## What it is

A **parallelism topology** for frontier-scale training that drops pipeline parallelism entirely. Standard PP (DualPipe, 1F1B, interleaved) accepts a *pipeline bubble* — idle time at the start/end of every microbatch cycle — in exchange for distributing layers across GPUs. HSDP's claim is that on dense enough interconnect you don't need PP: FSDP sharding scales to the parameter counts you care about, and the bubble you avoid is larger than the collectives you pay.

HSDP is built on a two-level cluster topology:

- **SuperPoD**: a group of nodes connected by a fast fabric (e.g., Ascend 910C SuperPoD with shared switch).
- **Across SuperPoDs**: slower inter-SuperPoD links.

Parameters are **FSDP-sharded within each SuperPoD** (fast collectives) and **replicated across SuperPoDs** (slow links carry only gradient reduction).

## How it works

```
within a SuperPoD (dp_shard=2, TP=4, EP=64, CP=32):
    layer i forward:    AllGather params_i          (intra-SuperPoD)
    layer i backward:   ReduceScatter grads_i       (intra-SuperPoD)

across SuperPoDs (dp_replicate=4):
    end of backward:    AllReduce grads             (inter-SuperPoD)
                        (fused, deferred, one shot)
```

Three systems tricks keep it fast:

**1. Zero-copy per-parameter collectives.** PyTorch-FSDP flattens parameters into one big buffer before AllGather, then copies them back out. HSDP runs AllGather/ReduceScatter on the individual parameter tensors in place — no flatten, no copy-in/out.

**2. Three-way backward overlap.** While layer *i* computes its backward:
   - layer *i+1*'s **ReduceScatter** is already in flight (intra-SuperPoD),
   - layer *i−1*'s **AllGather** for the next forward step is preparing,
   - the **inter-SuperPoD AllReduce** of gradients is fused and deferred to the end.

   Synchronization points are kept at the end of backward, not per-layer.

**3. Pre-allocated contiguous gradient buffer.** ReduceScatter writes gradient shards directly into their final slot; the fused inter-SuperPoD AllReduce reads contiguous memory without any packing step.

## Why it matters

On a 1024-NPU profiler with the 285B Xing 4 MoE at seq-len 262,144 and global batch 16 (TP4/EP64/CP32):

| Scheme | Step time |
|---|---|
| 1F1B PP (p=4, m=8) | 36.4 s |
| Interleaved PP (vpp=2) | 34.2 s |
| **HSDP (PP=1, dp_replicate=4, dp_shard=2)** | **31.8 s** |

HSDP is **1.14×** faster than 1F1B and **1.08×** faster than interleaved. The pipeline bubble at p=4, m=8 is 3/11 ≈ 27%; recovering most of that bubble dominates the zero-copy and overlap gains.

The deeper claim: on SuperPoD-dense fabric (and plausibly on H100 NVL72 / B200 clusters with NVLink-rich topologies), **pipeline parallelism is no longer load-bearing**. You get it back as system simplicity: no interleaved schedule, no micro-batch hyperparameter, no pipeline-stage balancing.

## Gotchas & tricks

- **Requires a real SuperPoD-class topology.** On nodes connected only by InfiniBand without a shared switch, intra-SuperPoD collectives aren't fast enough to beat 1F1B's bubble — HSDP degrades to plain FSDP and loses the advantage.
- **Scales as dp_shard × dp_replicate.** For truly massive models you still need enough parameter memory; the 285B / 17B-active Xing 4 fits with dp_shard=2. A trillion-parameter model would need dp_shard higher or a hybrid with TP absorbing more of the parameter weight.
- **Zero-copy collectives require a custom comm library.** The Ascend path uses MindSpore HyperParallel; the equivalent on CUDA would need NCCL extensions or Megatron-LM's communicator work.
- **No pipeline means no pipeline-based activation offload.** PP has a natural "offload activations between stages" story; HSDP has to rely on selective recomputation (which Xing 4 also does) or activation checkpointing.
- **Inter-SuperPoD AllReduce is the long pole.** Making it larger (dp_replicate=8) grows that collective; the sweet spot is around dp_replicate=4 at 1024-NPU scale.

## Sources

- Paper: *TeleChat to Xing4: Advancing Frontier Agentic Models on Domestic Chips and Frameworks* — TeleAI, 2026 — [arXiv:2611.22115](https://arxiv.org/abs/2611.22115)
- Related: standard FSDP / ZeRO-3 primary literature (Rajbhandari et al., FairScale/FSDP).
