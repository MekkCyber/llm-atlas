# Data Parallelism (DDP)

*Depth — the simplest form of parallelism: replicate the model, shard the batch, sum gradients.*

**TL;DR:** Every GPU holds a full copy of the model. The global batch is split across GPUs; each GPU computes forward/backward on its local shard; **gradients are all-reduced** before the optimizer step so every replica sees the same gradient. One all-reduce per step. Simplest parallelism, works out of the box, breaks the moment the model doesn't fit on one GPU. The foundation on top of which FSDP, TP, PP, and EP compose.

**Prereqs:** [_parallelism](_parallelism.md), [_communication-primitives](../systems/_communication-primitives.md)
**Related:** [fsdp](fsdp.md) · [tensor-parallelism](tensor-parallelism.md) · [pipeline-parallelism](pipeline-parallelism.md)

---

## What it is

Given N_d GPUs in a "data-parallel group":

- Each GPU holds an **identical copy** of the model parameters θ.
- Global batch of size B is split into **per-GPU shards** of size B/N_d.
- Each GPU independently computes forward + backward on its shard, producing a **local gradient g_i**.
- An **all-reduce** (sum or mean) combines the g_i across all N_d GPUs, so every GPU ends up with the same global gradient g = (1/N_d) · Σ_i g_i.
- Every GPU then runs **the same optimizer step** with the same gradient, so the parameters stay identical across replicas.

This is DistributedDataParallel (DDP) in PyTorch. NCCL is the usual collective backend.

Data parallelism **scales compute**: 2× more GPUs → 2× the tokens/second throughput (at the limit; bandwidth-bound at large N_d). It does **not scale memory**: the full model fits on each GPU, which is the whole constraint.

---

## How it works

### The training loop with DDP

```python
# One GPU's view
model = DistributedDataParallel(model, device_ids=[local_rank])
optimizer = AdamW(model.parameters(), lr=...)

for batch in dataloader:  # each rank gets a DIFFERENT shard of the batch
    optimizer.zero_grad()
    loss = model(batch).loss          # forward on local batch shard
    loss.backward()                    # local backward + gradient all-reduce (fused)
    optimizer.step()                   # identical step on all ranks
```

Key detail: the all-reduce is fused into `.backward()`. DDP registers gradient hooks on every parameter; as soon as a parameter's gradient is computed during backward, it's added to a bucket. When the bucket fills, DDP launches an all-reduce on the bucket **while the rest of the backward pass continues**. This **overlaps communication with backward computation** — a key practical speedup.

### The forward pass

Purely local. No communication. Each rank runs its batch shard through the full model locally. `B/N_d × S × H` input → `B/N_d × S × vocab` output.

### The backward pass

Two things happen in each layer's backward:

1. **Local gradient computation** — chain rule through the layer.
2. **All-reduce of the gradient** — as soon as a parameter's gradient tensor is finalized, it's queued into a bucket. Buckets flush via NCCL all-reduce.

Pseudocode for what DDP installs:

```python
def make_param_hook(bucket, idx):
    def hook(grad):
        bucket.add(idx, grad)
        if bucket.full():
            bucket.all_reduce_async()    # overlapped with next layer's backward
        return grad
    return hook

for i, p in enumerate(model.parameters()):
    bucket = get_bucket_for(p)
    p.register_hook(make_param_hook(bucket, i))
```

When `.backward()` returns, any pending all-reduces are synchronized (the `synchronize()` implicit at step end) so that the optimizer sees globally-averaged gradients.

### The optimizer step

Every replica has identical gradients after all-reduce, so the optimizer step is identical on every replica, so parameters stay in sync. No communication needed.

Initialization matters: weights must be **initialized identically on every rank** (the `DistributedDataParallel` constructor broadcasts rank-0's weights to all others as the first thing it does). Otherwise replicas drift.

### The key primitive: all-reduce

DDP fundamentally depends on **all-reduce sum**: every rank contributes a tensor, every rank ends up with the sum of all ranks' tensors. In modern NCCL this is implemented as **ring all-reduce**:

```
ring all-reduce = reduce-scatter + all-gather
```

On N_d ranks, each contributing a tensor of size `G` bytes:
- Phase 1 (reduce-scatter): each rank ends up with 1/N_d of the summed tensor.
- Phase 2 (all-gather): ranks exchange their reduced chunks until each has the full sum.
- Total bandwidth per rank: `2 · G · (N_d - 1) / N_d ≈ 2G` (at large N_d).

For DDP with a model of Ψ parameters (BF16 gradients → 2Ψ bytes):
- **Bytes per step per rank: 2 × 2Ψ = 4Ψ** (one all-reduce of 2Ψ-byte gradient tensor, both reduce-scatter and all-gather phases).

See [_communication-primitives](../systems/_communication-primitives.md) for the detailed ring algorithm.

### Bucketing

A 70B model has ~1,000 parameter tensors. Launching 1,000 small NCCL calls one per tensor would drown in launch overhead. DDP's **gradient bucketing** groups parameters into buckets (~25 MB default) and launches one all-reduce per bucket. Parameters are bucketed in **reverse order of computation** — last layer's gradients are ready first during backward, so they all-reduce while earlier layers are still computing. Perfect overlap is the design goal.

---

## Memory math

Per-GPU memory for a model of Ψ parameters under DDP with mixed-precision Adam:

| Thing | Size per GPU |
|---|---|
| FP16/BF16 params | 2Ψ |
| FP16/BF16 gradients | 2Ψ |
| FP32 optimizer states (Adam m + v) | 2 · 4Ψ = 8Ψ |
| FP32 master weights | 4Ψ |
| **Subtotal** | **16Ψ** |
| Activations (batch-dep) | O(B · S · H · L) |

A 7B model → **112 GB of model state per GPU** — doesn't fit on an 80 GB H100. This is precisely why FSDP exists.

DDP's memory doesn't reduce with more GPUs. Adding GPUs only adds batch/throughput.

---

## Why it matters

- **Simplest form that works.** Two lines of PyTorch. Scales to tens of thousands of GPUs if the model fits per-GPU.
- **Foundation of every other parallelism.** FSDP is "DDP but shard the state." TP/PP/EP/CP all compose with DDP as the outermost data-parallel loop. "DP group" as a concept appears in every frontier training stack.
- **No correctness gotchas.** The math is exactly equivalent to training on a single giant GPU. Gradient averaging is exact. Very few surprises compared to PP (bubbles, async) or TP (precision in all-reduce).
- **Comm-compute overlap is mature.** Bucket-based overlap means DDP has ~linear scaling to at least thousands of GPUs on modern fabrics.

---

## Gotchas & tricks

- **Learning rate scaling.** Global batch grows linearly with N_d. Typical convention: keep **per-GPU batch constant**, scale LR linearly (or by √N_d) as total batch grows. "Linear scaling rule" from the 1-hour ImageNet paper (Goyal 2017) is the baseline; LLMs sometimes use √-scaling.
- **Gradient accumulation composes.** If your per-GPU batch is still too big for memory, do K accumulation steps (forward/backward K times, no all-reduce until the K-th). `accumulate_grad_batches=K`. DDP handles this via `no_sync()` context manager — skip all-reduce for K-1 steps, do one big all-reduce on the final step.
- **`find_unused_parameters=False` by default.** If your model has parameters that aren't used in every forward (e.g., conditional branches, MoE gating), DDP needs `find_unused_parameters=True` — but this disables comm-compute overlap. Avoid unused parameters if you can.
- **SyncBatchNorm.** BatchNorm statistics are per-GPU by default in DDP — so BN effectively uses a batch of B/N_d, which breaks small-batch regimes. Replace with `SyncBatchNorm` which all-reduces BN statistics. Usually not an issue for Transformers (which use LayerNorm or RMSNorm).
- **DataLoader sampling.** Use `DistributedSampler` to ensure each rank sees a different shard of the global batch. Without it, every rank trains on identical data → effectively training a single model with N_d× redundancy.
- **Seed discipline.** Shuffle seeds must be coordinated (or explicitly differentiated) so the data shards don't overlap. DistributedSampler handles this.
- **NCCL_DEBUG=INFO is your friend.** DDP failures often manifest as NCCL hangs. Enable NCCL debug to see the collective each rank is stuck on.
- **Watch out for dropped ranks.** If one rank dies or hangs, all-reduce blocks forever on the other ranks. Set `NCCL_TIMEOUT` and have a watchdog restart the job cleanly.
- **Comm-compute overlap requires backward to take time.** For very small models where backward is fast, the all-reduce can't be fully overlapped — DDP throughput flattens. At that point you're bandwidth-bound, not compute-bound.
- **Hierarchical all-reduce for large clusters.** NCCL auto-picks a ring-based or tree-based all-reduce depending on topology. For 10K+ GPU clusters, a two-level (intra-node + inter-node) pattern may outperform a flat ring. Rarely needs manual tuning.

---

## DDP vs FSDP: when to switch

Rule of thumb: **if 16Ψ (model state) ≤ per-GPU memory** minus your activation footprint, use DDP. Otherwise use FSDP. For a 7B model on H100 (80 GB), activations dominate; you can sometimes fit DDP with careful activation checkpointing. For 70B, you cannot — FSDP (or FSDP + TP) is mandatory.

---

## Sources

- Paper: *PyTorch Distributed: Experiences on Accelerating Data Parallel Training* — Li et al., 2020, arXiv 2006.15704 — canonical description of DDP's bucketing + overlap algorithm.
- Paper: *Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour* — Goyal et al., 2017, arXiv 1706.02677 — linear LR scaling rule for large-batch DDP training.
- Paper: *Horovod: fast and easy distributed deep learning in TensorFlow* — Sergeev & Del Balso, 2018, arXiv 1802.05799 — the ring-allreduce-based DDP that predates PyTorch DDP.
- Docs: PyTorch DistributedDataParallel — https://pytorch.org/docs/stable/notes/ddp.html.
- Docs: NCCL — https://docs.nvidia.com/deeplearning/nccl/.
