# Parallelism for Large-Model Training

*Taxonomy — the ways we split a model and its computation across many GPUs.*

**TL;DR:** A trillion-parameter model does not fit on one GPU; its forward pass does not fit on one GPU even when sharded; and even if it did, training it in a year requires 10⁴–10⁵ accelerators cooperating. The way we get there is **parallelism**: slicing the model, the data, or the sequence along some axis and coordinating the pieces. Five canonical axes — **data, tensor, pipeline, context, expert** — plus a sequence-sharding variant sit on top of one shared communication substrate (NCCL collectives on top of NVLink/RoCE/IB). Modern frontier runs compose 4 or 5 of these simultaneously (e.g., Llama 3's `[TP, CP, PP, DP]` with FSDP as the DP mechanism).

**Related taxonomies:** [_communication-primitives](../systems/_communication-primitives.md)
**Depth files covered here:** [data-parallelism](data-parallelism.md) · [fsdp](fsdp.md) · [tensor-parallelism](tensor-parallelism.md) · [pipeline-parallelism](pipeline-parallelism.md) · [context-parallelism](context-parallelism.md) · [sequence-parallelism](sequence-parallelism.md) · [expert-parallelism](expert-parallelism.md)

---

## The problem

A dense Transformer at scale has four things that must live somewhere:

1. **Parameters** (weights): Ψ = O(N) bytes. For a 405B model in BF16, Ψ ≈ 810 GB — 10× the memory of an H100.
2. **Gradients**: one per parameter, same size as parameters during a step.
3. **Optimizer states**: Adam keeps momentum + variance, each same size as parameters, typically in FP32. For 405B with mixed-precision Adam, optimizer states alone ≈ 6.5 TB.
4. **Activations**: intermediate tensors from the forward pass, needed for the backward. For batch B, sequence S, hidden H, L layers: roughly `B · S · H · L · k` bytes where k depends on architecture. At S = 128k this dominates all of the above.

Plus a fifth: **compute**. A single H100 at 1000 TFLOP/s of BF16 finishes a 405B forward+backward step in ~O(10³) seconds of compute; you need 10⁴ of them for the step to complete in ~0.1 s of wall clock.

**No single form of parallelism solves all of this.** Data parallelism replicates the model and shards the batch — good for compute, useless for memory. Tensor parallelism shards each weight matrix — good for memory of one layer, doesn't scale past ~8 (NVLink domain). Pipeline parallelism puts different layers on different GPUs — good for memory, leaves bubbles. Context parallelism shards the sequence — needed for 128k. Expert parallelism puts different MoE experts on different GPUs — specific to MoE architectures.

Every real run composes multiple axes.

---

## The shared pattern

Every parallelism scheme answers four questions:

1. **What gets split?** Batch dimension, parameter tensors, layers, sequence dimension, or experts.
2. **What communication primitive closes the gap?** All-reduce, all-gather, reduce-scatter, point-to-point, or all-to-all. See [_communication-primitives](../systems/_communication-primitives.md).
3. **How often does that primitive fire?** Every gradient step, every layer forward, every attention computation, or every micro-batch boundary.
4. **What's the ratio of compute to communication?** If communication is cheap relative to compute, the parallelism scales linearly; if not, you hit a wall.

Matrix of where each primitive lives:

| Axis | Splits | Primitive | When it fires |
|---|---|---|---|
| [Data](data-parallelism.md) | Batch | **all-reduce** of gradients | Once per gradient step |
| [FSDP](fsdp.md) (ZeRO-3 data parallel) | Params + grads + opt-states across the DP group | **all-gather** (fwd+bwd) + **reduce-scatter** (grads) | Once per wrap-unit per fwd/bwd |
| [Tensor](tensor-parallelism.md) | Within a weight matrix | **all-reduce** | Twice per transformer layer |
| [Pipeline](pipeline-parallelism.md) | Across layers | **point-to-point** | Once per micro-batch boundary |
| [Context](context-parallelism.md) | Sequence dimension | **all-gather** (Llama 3) or **ring SendRecv** (Liu 2023) | Once per attention layer |
| [Sequence (Megatron-SP)](sequence-parallelism.md) | Seq dim for LN/Dropout | **all-gather** + **reduce-scatter** | In place of TP's all-reduce |
| [Expert](expert-parallelism.md) | MoE experts | **all-to-all** (dispatch + combine) | Twice per MoE layer |

---

## Variants

| Technique | Splits what | Main cost | Scales to | When it wins |
|---|---|---|---|---|
| [Data parallelism](data-parallelism.md) | Batch only; full model replicated | 1 all-reduce of gradients per step | Bandwidth-bound at large N_d | Small-to-medium models that fit on one GPU |
| [FSDP / ZeRO](fsdp.md) | Params + grads + opt-states across DP group | 1.5× DDP comm (ZeRO-3) | Model-memory-bound only by slowest device's local shard | Default when model doesn't fit per-GPU |
| [Tensor parallelism](tensor-parallelism.md) | Within a weight matrix, across heads/columns | 2 all-reduces per layer | **Within a server (NVLink, ≤8–16 GPUs)** | Reduces per-GPU param + activation memory; essential companion to PP |
| [Pipeline parallelism](pipeline-parallelism.md) | Model into layer-contiguous stages | Point-to-point at stage boundaries | Up to ~16–32 stages before bubble eats throughput | Very deep models, off-server scaling |
| [Context parallelism](context-parallelism.md) | Sequence into chunks across ranks | All-gather K,V or ring-rotate K,V per attention | Scales linearly with seq len | 128k+ context; GQA makes it cheap |
| [Sequence parallelism (Megatron-SP)](sequence-parallelism.md) | LN/Dropout activations along seq | Replaces TP's all-reduce with all-gather + reduce-scatter | Same scale as TP | Pairs with TP; reduces TP's replicated activation memory |
| [Expert parallelism](expert-parallelism.md) | MoE experts across ranks | 2 all-to-alls per MoE layer | Scales with #experts | Any MoE — mandatory for large expert counts |

### Common compositions

| Recipe | Use case |
|---|---|
| DDP only | Small model that fits per-GPU |
| FSDP only | Medium dense model (≤70B), can afford the extra all-gather |
| TP × PP × DP | Classical Megatron-3D for dense frontier models |
| TP × CP × PP × FSDP | **Llama 3's 4D** for 405B dense at 128k context |
| TP × PP × EP × DP | DeepSeek-V3-style MoE with MoE-all-to-all on top of 3D |
| FSDP + TP only | Recent ergonomic default for ≤70B (no pipeline bubble) |

---

## How to choose

1. **Does the model fit on one GPU** (params + grads + opt-states + one layer's activations)? If yes — **DDP**. Done. Do not add complexity.
2. **If not, can FSDP (ZeRO-3) close the gap alone**? Each rank holds ~Ψ/N_d bytes of params + grads + opt-states. Plus O(1 layer) of reconstructed full weights during fwd/bwd. For a **≤70B model on ≥64 GPUs**, usually yes.
3. **If FSDP is not enough** (per-layer activations still too big, or the unit-wise all-gather cost is too high), **add TP within the NVLink domain**. TP=8 is the usual ceiling (one server, NVLink).
4. **If that's still not enough, add PP across servers**. PP is expensive per-bubble but cheap per-communication (only point-to-point at stage boundaries).
5. **For long context (>32k)**, add **CP**. All-gather CP (Llama 3) when K/V are small relative to Q (GQA/MQA); ring CP (Liu 2023) otherwise.
6. **For MoE**, add **EP**. Usually on top of TP × PP × DP.

The order of composition matters for topology awareness: put the most-communicating axes on the fastest links. TP > CP > PP > DP in terms of comm frequency → inside-out: `[TP, CP, PP, DP]` puts TP innermost (within NVLink), DP outermost (tolerates latency via FSDP overlap).

---

## Compute/communication tradeoff

A useful mental model: compute per layer is `O(B · S · H²)`; communication per layer depends on the scheme:

| Scheme | Comm per layer (bytes) | Ratio vs compute |
|---|---|---|
| DDP | `O(H²) / step` (amortized over many layers) | Tiny |
| FSDP (ZeRO-3) | `O(H²)` per layer | Medium; overlapped with compute via prefetch |
| TP | `O(B · S · H)` per all-reduce, 2 per layer | Large; hidden only by NVLink's 600 GB/s+ |
| PP | `O(B · S · H)` per stage boundary, point-to-point | Small per transfer; bubbles are the real cost |
| CP (ring) | `O(B · S · H_KV / N_CP)` per iteration, N_CP iterations | Overlapped with blockwise compute |
| CP (all-gather) | `O(B · S · H_KV)` once per layer | Small when KV heads << query heads (GQA) |
| EP | `O(B · S · H)` per all-to-all, 2 per MoE layer | Expensive; requires fast fabric |

The **arithmetic intensity** of each primitive (compute per byte communicated) determines whether it scales. TP within a server has arithmetic intensity of `O(H)` — fine on NVLink, terrible on Ethernet. PP has point-to-point transfers of `O(B · S · H)` with `O(B · S · H² / PP)` compute per stage — arithmetic intensity grows with H, which is why PP scales better than TP cross-server.

---

## Adjacent but distinct

- **Gradient accumulation.** Running multiple micro-batches before a step, summing gradients locally. Pure memory trick, no inter-GPU communication. Composes with every parallelism above.
- **Activation checkpointing** (recomputation). Drop activations during forward, recompute during backward. Pure memory/compute tradeoff; not parallelism. But interacts: more aggressive checkpointing enables larger micro-batch counts, which shrinks pipeline bubbles. See [pipeline-parallelism](pipeline-parallelism.md).
- **Offloading** (ZeRO-Infinity, CPU offload). Park params/grads/opt-states in CPU or NVMe. Not parallelism per se; a memory hierarchy trick that extends FSDP.
- **ZeRO++, 2D hybrid sharding.** Shard the FSDP all-gather within node + replicate across nodes. Reduces inter-node bandwidth at the cost of intra-node memory. Llama 3 uses this form implicitly with their "hybrid sharding" (FSDP Section 3.4).

---

## Sources

- Paper: *Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism* — Shoeybi et al., 2019, arXiv 1909.08053 — tensor parallelism.
- Paper: *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models* — Rajbhandari et al., 2020, arXiv 1910.02054 — ZeRO stages, the FSDP foundation.
- Paper: *PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel* — Zhao et al., 2023, arXiv 2304.11277 — FlatParameter, prefetching, hybrid sharding.
- Paper: *GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism* — Huang et al., 2019, arXiv 1811.06965 — pipeline parallelism, bubble formula.
- Paper: *PipeDream: Fast and Efficient Pipeline Parallel DNN Training* — Harlap et al., 2018, arXiv 1806.03377 — 1F1B scheduling.
- Paper: *Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM* — Narayanan et al., 2021, arXiv 2104.04473 — interleaved 1F1B, 3D parallelism.
- Paper: *Ring Attention with Blockwise Transformers for Near-Infinite Context* — Liu et al., 2023, arXiv 2310.01889 — ring-based context parallelism.
- Paper: *Reducing Activation Recomputation in Large Transformer Models* — Korthikanti et al., 2022, arXiv 2205.05198 — Megatron sequence parallelism + selective recomputation.
- Paper: *The Llama 3 Herd of Models* — Grattafiori et al., 2024, arXiv 2407.21783 — 4D parallelism with all-gather CP.
- Paper: *GShard* — Lepikhin et al., 2020, arXiv 2006.16668 — expert parallelism via all-to-all.
