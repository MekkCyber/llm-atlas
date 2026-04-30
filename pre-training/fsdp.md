# FSDP / ZeRO (Fully Sharded Data Parallelism)

*Depth — shard the model state across the data-parallel group, gather on demand, so each GPU holds only Ψ/N_d of the optimizer state.*

**TL;DR:** DDP replicates the full model on every GPU; FSDP **shards params + gradients + optimizer states across the data-parallel group**, reconstructing full weights on demand by **all-gather** before each layer's forward/backward and **reduce-scattering** gradients after backward. Memory per GPU drops from `16Ψ` to `16Ψ / N_d`. Communication grows from `2Ψ` to `3Ψ` per step — 1.5× DDP — but prefetching hides most of it. Introduced as **ZeRO** (Rajbhandari 2020), productized as PyTorch **FSDP** (Zhao 2023). The default "can't fit on one GPU" path.

**Prereqs:** [data-parallelism](data-parallelism.md), [_communication-primitives](../systems/_communication-primitives.md)
**Related:** [_parallelism](_parallelism.md) · [tensor-parallelism](tensor-parallelism.md) · [pipeline-parallelism](pipeline-parallelism.md)

---

## What it is

A data-parallel scheme that trades extra communication for linear memory scaling. Instead of replicating everything across N_d ranks, **shard** the parts of training state that are only needed *sometimes*.

ZeRO defines three stages (Rajbhandari 2020):

| Stage | Shards | Per-GPU memory | Comm vs DDP |
|---|---|---|---|
| **ZeRO-1 (P_os)** | Optimizer states only | `4Ψ + KΨ/N_d` ≈ **4Ψ** | Same |
| **ZeRO-2 (P_os+g)** | Optimizer states + gradients | `2Ψ + (2Ψ + KΨ)/N_d` ≈ **2Ψ** | Same |
| **ZeRO-3 (P_os+g+p)** | Optimizer states + gradients + params | **`16Ψ / N_d`** | **1.5×** DDP |

(`K = 12` for mixed-precision Adam: FP32 master params 4 bytes + momentum 4 + variance 4.)

PyTorch **FSDP** (FullyShardedDataParallel) implements ZeRO-3. Other frameworks (DeepSpeed, Megatron-FSDP, TorchTitan) offer variants of ZeRO-1/2/3 with different ergonomics.

FSDP is the standard answer to "my model doesn't fit on one GPU." For a 70B model on 64 H100s, ZeRO-3 brings per-GPU model state down from 1120 GB to 17.5 GB — comfortable.

---

## How it works

### The sharding unit — FlatParameter

FSDP groups parameters into "wrap units" — a handful of contiguous layers, typically. Each unit's parameters are **flattened and concatenated** into a single `FlatParameter` tensor, then **sharded equally across the data-parallel group**. Rank `r` holds **bytes `r·Ψ_unit/N_d` to `(r+1)·Ψ_unit/N_d`** of the FlatParameter.

```python
# Conceptually
class FlatParameter(nn.Parameter):
    def __init__(self, params, world_size, rank):
        flat = torch.cat([p.detach().flatten() for p in params])  # all params in one 1D tensor
        shard_size = flat.numel() // world_size
        start = rank * shard_size
        end = start + shard_size
        super().__init__(flat[start:end].clone())  # only MY shard
```

Reality has more bookkeeping — FSDP knows which slice of the FlatParameter corresponds to which original parameter and which module — but the core idea is "flatten, shard, reconstruct on demand."

### The forward pass

Per wrap unit, during forward:

```
1. AllGather FlatParameter       → every rank now holds full params for this unit
2. Compute forward on full params
3. Drop the non-owned shards     → back to Ψ_unit/N_d
4. Cache activations (as usual)
```

Code-level:

```python
def fsdp_forward(self, x):
    # Phase 1: gather
    full_params = all_gather(self.local_shard, group=self.dp_group)  # NCCL AllGather
    set_params(self.module, full_params)

    # Phase 2: compute
    y = self.module(x)

    # Phase 3: drop
    free(full_params)
    return y
```

The all-gather is per-layer (per wrap unit). Cost: each rank contributes its shard; each rank ends up with the full Ψ_unit worth of bytes. Total data movement per rank over a full forward pass: `Ψ · (N_d − 1)/N_d ≈ Ψ` bytes.

### The backward pass

Per wrap unit, during backward (in reverse order):

```
1. AllGather FlatParameter       → full params again (same all-gather cost)
2. Backward compute: compute full-param gradients using cached activations
3. ReduceScatter gradients       → each rank keeps 1/N_d of the summed gradients
4. Drop full params + full gradients
```

Code:

```python
def fsdp_backward(self, grad_out):
    # Phase 1: gather (same as forward)
    full_params = all_gather(self.local_shard, group=self.dp_group)
    set_params(self.module, full_params)

    # Phase 2: backward compute
    full_grads = backward(self.module, grad_out)

    # Phase 3: reduce-scatter gradients (rank r ends up with 1/N_d of the sum)
    self.local_grad = reduce_scatter(full_grads, group=self.dp_group)  # NCCL ReduceScatter

    # Phase 4: drop
    free(full_params); free(full_grads)
```

Costs per rank per backward:
- **AllGather**: `Ψ · (N_d-1)/N_d ≈ Ψ`.
- **ReduceScatter**: `Ψ · (N_d-1)/N_d ≈ Ψ`.

### Total communication

Per rank per step:

- Forward: `Ψ` (AllGather).
- Backward: `Ψ` (AllGather) + `Ψ` (ReduceScatter) = `2Ψ`.
- **Total: 3Ψ** vs DDP's `2Ψ`.

That's the **1.5× DDP** cost Rajbhandari 2020 reports for ZeRO-3 (§7.2.2). ZeRO-1 and ZeRO-2 preserve DDP comm cost (`2Ψ`) because their shard reconstruction can be merged with the existing all-reduce.

### Optimizer step

Each rank holds only its shard of the optimizer states and applies Adam/AdamW to its own shard of the gradients. **Zero communication** at optimizer step. Result: each rank updates its shard of the parameters in place; the next forward's all-gather reconstructs the full updated params.

```python
def fsdp_optimizer_step(self):
    # All tensors below are 1/N_d sized
    m = beta1 * m + (1 - beta1) * self.local_grad
    v = beta2 * v + (1 - beta2) * self.local_grad ** 2
    self.local_shard -= lr * m / (sqrt(v) + eps)
```

### Overlap via prefetching

The naïve pattern — "gather, compute, drop, gather next, compute next" — has zero communication/compute overlap. FSDP fixes this with **prefetching**:

- **Forward prefetch**: while layer k is computing, issue the all-gather for layer k+1 on a separate CUDA stream.
- **Backward prefetch**: while layer k's backward is computing, issue the all-gather for layer k-1's backward.

```python
# Simplified FSDP forward with prefetch
def fsdp_forward_prefetched(self, x):
    layer = 0
    next_gather = all_gather_async(layers[0].local_shard)

    for layer in range(num_layers):
        full_params_l = next_gather.wait()                    # synchronize current layer's gather
        if layer + 1 < num_layers:
            next_gather = all_gather_async(layers[layer+1].local_shard)  # fire next layer's gather
        y = layers[layer](x, full_params_l)                   # compute with current full params
        free(full_params_l)
        x = y
    return x
```

The backward prefetch is symmetric. Whether the overlap fully hides communication depends on (compute/comm) ratio per layer.

### Hybrid sharding — 2D FSDP

At very large N_d, the all-gather bandwidth becomes the bottleneck. **Hybrid sharding** splits the DP group into:
- An **intra-node shard group** (e.g., 8 GPUs within a server with NVLink).
- An **inter-node replica group** (across nodes with slower RoCE/IB).

Params are fully sharded within the intra-node group (so each node has one full copy); gradients are all-reduced across the inter-node replica group (DDP-style). This cuts inter-node communication at the cost of intra-node memory. Used by Llama 3 and most 2024+ frontier runs.

```python
# Conceptually:
intra_node_group = GPUs on the same server  # fast NVLink
inter_node_group = one GPU per server       # slower RoCE

# Intra-node: full shard (ZeRO-3 style)
# Inter-node: DDP (replicate within node across nodes)
```

Llama 3 Sec. 3.3.2: *"for model shards we do not reshard after forward computation to avoid an extra all-gather communication during backward passes"* — effectively a ZeRO-2-ish variant within the intra-node shard, a further optimization on hybrid sharding.

---

## Memory math

Before (DDP) vs after (ZeRO-3) per GPU, for Ψ params:

| Item | DDP | ZeRO-3 |
|---|---|---|
| Params (BF16) | 2Ψ | 2Ψ / N_d (+ Ψ for the active unit during fwd/bwd) |
| Gradients (BF16) | 2Ψ | 2Ψ / N_d |
| Optimizer states (FP32 × 2 + FP32 master + extras = KΨ, K=12) | 12Ψ | 12Ψ / N_d |
| **Subtotal** | **16Ψ** | **16Ψ / N_d + Ψ_unit** |

For N_d = 64 and Ψ = 70B:
- DDP: 16 × 70 GB = **1120 GB per GPU** (doesn't fit).
- ZeRO-3: 16 × 70 / 64 ≈ 17.5 GB + Ψ_unit ≈ few GB per GPU.

That's the whole story — FSDP enables training 70B on 64 H100s where DDP cannot.

---

## Why it matters

- **Memory-linear-in-N_d scaling** for model state. Adding GPUs directly buys you memory headroom, not just throughput.
- **Drop-in replacement for DDP.** One line of PyTorch change (`DDP(...)` → `FSDP(...)`). No changes to the model.
- **Composes with TP, PP, CP, EP.** FSDP is orthogonal — it can wrap any model and shard the DP dimension.
- **Baseline for modern frontier training.** Llama 3, OLMo 2, most 2024+ dense models use FSDP as the DP mechanism.
- **Activation sharing mostly unaffected.** FSDP only shards *parameters and their gradients*; activations are still per-rank (sharded by batch, not by FSDP). For activation sharding you need TP or CP.

---

## Gotchas & tricks

- **Wrap granularity matters.** Wrap each Transformer block as one FSDP unit. Finer → more frequent all-gathers (overhead). Coarser → larger peak memory during a unit's forward. Default for Transformers: one wrap unit per decoder layer.
- **`use_orig_params=True`** (newer FSDP flag). Preserves PyTorch's usual parameter semantics; without it, accessing `model.layer.weight` returns the FlatParameter slice, which breaks many optimizers and debuggers.
- **Gradient accumulation without `no_sync()`.** FSDP does not yet have a perfect `no_sync()` — accumulating without sync still triggers reduce-scatter. For proper accumulation use the `sync_module_states=False` pattern or accumulate at a higher level.
- **Optimizer state sharding is auto.** FSDP hands each rank only its shard of gradients → the optimizer naturally creates only-its-shard of optimizer states. No manual sharding needed.
- **Mixed precision is tricky.** FSDP has explicit `MixedPrecision` config for param dtype / gradient dtype / buffer dtype. Getting it right matters — wrong choice can lead to FP32 params materializing during all-gather.
- **`limit_all_gathers=True`** is often needed at very large scale — caps the number of concurrent in-flight all-gathers so peak memory doesn't balloon.
- **Checkpointing is more complex.** Standard `state_dict()` returns the sharded state. For resume or inference you need `FullStateDictConfig` to gather the full model. Distributed checkpointing (`torch.distributed.checkpoint`) is the modern answer.
- **Communication vs overlap tuning.** If compute per layer is small relative to all-gather time, FSDP throughput drops. Solution: larger micro-batch (more compute per layer) or TP inside each layer to reduce per-GPU all-gather volume.
- **FSDP + TP**: FSDP shards DP; TP shards within a layer. Orthogonal. At frontier scale you run both — FSDP across nodes, TP within a node.
- **Prefetch depth.** FSDP's `backward_prefetch=BackwardPrefetch.BACKWARD_PRE` issues the next all-gather *before* the current backward's reduce-scatter completes. `BACKWARD_POST` is safer but less overlap. Choose based on compute vs comm ratio.
- **`reshard_after_forward=False`** (advanced). Keep the all-gathered params in memory after forward so you don't all-gather again during backward. Saves comm (matches Llama 3's choice). Costs per-unit memory = Ψ_unit instead of Ψ_unit/N_d.
- **ZeRO-Offload / ZeRO-Infinity.** DeepSpeed variants that offload optimizer states and/or parameters to CPU or NVMe. Not "pure" ZeRO-3 anymore; enables training models far larger than aggregate GPU memory at the cost of throughput. Rarely used for frontier dense models.

---

## Sources

- Paper: *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models* — Rajbhandari et al., SC 2020, arXiv 1910.02054 — the three stages + memory + communication analysis.
- Paper: *PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel* — Zhao et al., 2023, arXiv 2304.11277 — FlatParameter, prefetching, hybrid sharding.
- Paper: *ZeRO-Offload: Democratizing Billion-Scale Model Training* — Ren et al., 2021, arXiv 2101.06840 — CPU offload variant.
- Paper: *ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning* — Rajbhandari et al., 2021, arXiv 2104.07857 — NVMe offload + scaling to trillion-param training.
- Docs: PyTorch FSDP — https://pytorch.org/docs/stable/fsdp.html.
- Paper: *The Llama 3 Herd of Models* — Meta, 2024 — uses FSDP with `reshard_after_forward=False` tweak + hybrid sharding.
