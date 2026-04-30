# Tensor Parallelism (TP / Megatron-LM)

*Depth — split every weight matrix across GPUs within a layer; all-reduce partial sums before the next layer.*

**TL;DR:** Take a weight matrix `W ∈ ℝ^(in×out)`, split it along rows or columns across N_t GPUs, have each GPU compute part of the matmul on the full input, and **all-reduce the partial results** to get the full output. Two all-reduces per Transformer block (one for attention, one for MLP). Designed for intra-server NVLink — scales to ~8 GPUs per tensor-parallel group, doesn't scale beyond. The foundation of Megatron-LM. Essential companion to PP when a single layer's activations don't fit on one GPU.

**Prereqs:** [_parallelism](_parallelism.md), [_communication-primitives](../systems/_communication-primitives.md), [attention](../fundamentals/attention.md)
**Related:** [fsdp](fsdp.md) · [pipeline-parallelism](pipeline-parallelism.md) · [sequence-parallelism](sequence-parallelism.md)

---

## What it is

Instead of replicating a weight matrix on every GPU (DDP) or sharding whole layers across GPUs (PP), TP shards **individual matrix multiplications**. Every GPU in the TP group participates in every matmul, each computing a slice of the output.

Intuition: a big matmul `Y = X W` can be partitioned along `W`'s columns:
- `W = [W₁, W₂]` (column split).
- `Y = X W = [X W₁, X W₂] = [Y₁, Y₂]`.
- GPU 1 computes Y₁ = X W₁; GPU 2 computes Y₂ = X W₂.
- Concat Y₁, Y₂ if the next op needs the full Y.

Or along `W`'s rows (requires splitting `X` too):
- `W = [W₁; W₂]` (row split), `X = [X₁, X₂]`.
- `Y = X W = X₁ W₁ + X₂ W₂` → each GPU computes one partial sum; **all-reduce** the partials.

Megatron-LM (Shoeybi 2019) chains these two patterns so each Transformer block does exactly **one all-reduce** per forward sub-block (attention or MLP) by design — the partial sums from the row-parallel GEMM are summed into the input of the next layer without an extra gather.

---

## How it works

### The MLP block

A standard MLP in a Transformer is:

```
Y = GeLU(X A) · B
```

where `X ∈ ℝ^(B·S × H)`, `A ∈ ℝ^(H × 4H)`, `B ∈ ℝ^(4H × H)`, `Y ∈ ℝ^(B·S × H)`.

**TP strategy** (Megatron Sec. 3, Figure 3a):

1. **A is column-parallel.** Split `A = [A₁, A₂]` along output columns. Each of N_t GPUs holds `A_i ∈ ℝ^(H × 4H/N_t)`.
2. Each GPU computes locally: `Y_i^{pre} = GeLU(X A_i) ∈ ℝ^(B·S × 4H/N_t)`.
3. **B is row-parallel.** Split `B = [B₁; B₂]` along input rows. Each of N_t GPUs holds `B_i ∈ ℝ^(4H/N_t × H)`.
4. Each GPU computes locally: `Y_i = Y_i^{pre} B_i ∈ ℝ^(B·S × H)` — a partial sum.
5. **All-reduce across the TP group** to produce `Y = Σ_i Y_i`.
6. Dropout + residual (the residual comes from the pre-TP input, which was replicated on every GPU — so nothing to sum).

Key property: **GeLU is element-wise**. If `A` is column-parallel, each GPU's partial output `Y_i^{pre}` is independently a valid input to GeLU — no cross-GPU communication needed to apply GeLU. This is what makes the "column-then-row" split clean.

```python
# Simplified Megatron MLP forward
class TPColumnLinear(nn.Module):
    def __init__(self, in_features, out_features, tp_group):
        self.A_shard = nn.Parameter(torch.empty(in_features, out_features // tp_world_size))

    def forward(self, x):
        # x is replicated across TP group (duplicated input)
        return x @ self.A_shard                    # local matmul, no comm

class TPRowLinear(nn.Module):
    def __init__(self, in_features, out_features, tp_group):
        self.B_shard = nn.Parameter(torch.empty(in_features // tp_world_size, out_features))

    def forward(self, x_sharded):
        # x_sharded is column-sharded across TP group
        y_partial = x_sharded @ self.B_shard        # local matmul
        y = all_reduce(y_partial, group=tp_group)   # sum partial sums across TP group
        return y

def tp_mlp(x):
    y1 = TPColumnLinear_A(x)          # no comm
    y1 = gelu(y1)                     # no comm (element-wise)
    y2 = TPRowLinear_B(y1)            # ONE all-reduce
    return y2
```

### The attention block

Self-attention: `Q = X W_Q; K = X W_K; V = X W_V; out = softmax(Q K^T / √d) V · W_O`.

**TP strategy** (Megatron Sec. 3, Figure 3b):

1. **Q, K, V projections are column-parallel along the head dimension.** If the attention has H_total heads and `W_Q ∈ ℝ^(d_model × (H_total · d_head))`, split the head dimension: `W_Q = [W_Q¹, W_Q²]`. GPU 1 gets heads 0..H_total/2 − 1; GPU 2 gets heads H_total/2..H_total − 1.
2. Each GPU computes its own `Q_i, K_i, V_i` and its own `attention_i = softmax(Q_i K_i^T / √d) V_i` **using only its local heads**. No cross-GPU communication during attention — heads are independent.
3. **Output projection W_O is row-parallel.** Each GPU multiplies its `attention_i` by its slice of `W_O`, producing a partial sum.
4. **All-reduce** across the TP group.

```python
class TPSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, tp_world_size, tp_group):
        self.heads_per_gpu = n_heads // tp_world_size
        self.d_head = d_model // n_heads

        # Column-parallel QKV: local slice covers heads_per_gpu heads
        self.W_QKV = nn.Parameter(torch.empty(d_model, 3 * self.heads_per_gpu * self.d_head))
        # Row-parallel output proj
        self.W_O = nn.Parameter(torch.empty(self.heads_per_gpu * self.d_head, d_model))

    def forward(self, x):
        qkv = x @ self.W_QKV                            # [B,S, 3*heads_per_gpu*d_head]
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, S, self.heads_per_gpu, self.d_head).transpose(1, 2)
        # ... same for k, v
        out_local = scaled_dot_product_attention(q, k, v)      # per-head, local
        out_local = out_local.transpose(1, 2).reshape(B, S, -1)

        out = out_local @ self.W_O                       # local row-parallel matmul (partial sum)
        out = all_reduce(out, group=self.tp_group)       # ONE all-reduce
        return out
```

### The f and g conjugate operators

Megatron's implementation trick: define two dual operators **f** and **g** that abstract the synchronization:

- **f**: identity in forward, all-reduce in backward. Placed at **input** of the column-parallel GEMM.
- **g**: all-reduce in forward, identity in backward. Placed at **output** of the row-parallel GEMM.

Why: the residual connection input must see the *full* gradient — but when backward propagates through the row-parallel GEMM, each rank only has its local gradient. An all-reduce upstream (in backward, at f) collects the gradient.

```python
class f(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x               # identity
    @staticmethod
    def backward(ctx, grad_out):
        return all_reduce(grad_out)  # sum gradients across TP group

class g(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return all_reduce(x)   # sum partial sums
    @staticmethod
    def backward(ctx, grad_out):
        return grad_out        # identity

def tp_mlp_with_operators(x):
    x = f(x)                   # identity forward, all-reduce backward
    y1 = x @ A_shard           # column-parallel
    y1 = gelu(y1)
    y2 = y1 @ B_shard          # row-parallel — partial sum
    y2 = g(y2)                 # all-reduce forward, identity backward
    return y2
```

**Two all-reduces per Transformer layer per forward pass** (one at MLP's g, one at attention's g), and **two all-reduces per backward pass** (at f's of MLP and attention). Total = **4 all-reduces per layer per step**.

### Communication cost

Per all-reduce: each rank exchanges `B · S · H` bytes (the input size). With 4 all-reduces per layer × L layers × BF16 (2 bytes):

```
bytes per step per rank = 4 · L · B · S · H · 2 = 8 · B · S · H · L   bytes
```

For a 70B model (L=80, H=8192) at B=4, S=8192:
- Per all-reduce: ~500 MB.
- Per step: ~40 GB of communication per rank. At NVLink 600 GB/s → ~67 ms of comm.

This is why TP is **NVLink-only**. Cross-server (RoCE at 400 Gbps) the same volume takes 800 ms — lethal.

### TP within a server only

TP = 8 is the usual ceiling: 8 GPUs per DGX server, all on one NVLink fabric at ~600+ GB/s. TP > 8 forces cross-server all-reduces — throughput collapses.

---

## Memory effect

TP shards:
- **Weights**: `A, B` weights split N_t-ways → params/GPU `= Ψ / N_t` for the TP-sharded portion.
- **Activations inside the block**: the GeLU-output `Y^{pre}` is `B · S · 4H/N_t` per rank — N_t× smaller than a replicated version.

But TP does **NOT** shard:
- **Layer norms**, dropout, residual streams (replicated across TP group). Megatron-SP later addresses this.
- **Embedding weights and LM head** (usually not TP-parallelized, or sharded separately).

Net memory per GPU under TP=t:
- Block params: `Ψ_block / t`.
- Block activations: `B · S · H · (factor / t)` for the TP-internal tensors, full `B · S · H` for the TP-external ones (LN, residual).

That "full B · S · H for LN/residual" is why [sequence-parallelism](sequence-parallelism.md) exists — it shards those along seq dim.

---

## Why it matters

- **Memory inside a layer.** When a single layer's activations don't fit per-GPU (large H, large S), TP is the only way to shrink them — FSDP doesn't help (FSDP shards params, not per-layer activations).
- **Companion to PP.** PP distributes *different* layers across GPUs; within each stage you still need each layer to fit. TP handles that.
- **Mature kernels.** Flash Attention, fused QKV, fused MLP all compose with TP. Megatron's implementation is battle-tested at 500B+ scale.
- **The bottleneck is NVLink.** TP's scaling ceiling is fundamentally the intra-server interconnect bandwidth. As NVSwitch improves (H100's 900 GB/s, B200's 1.8 TB/s), TP ceilings rise.

---

## Gotchas & tricks

- **TP = 8 is the practical ceiling.** With NVSwitch-connected 8×H100 or 8×A100, TP=8 sits comfortably on one server. TP=4 or TP=2 is common for smaller models.
- **Don't cross the server.** TP across servers (via RoCE/IB) destroys throughput. If you need more parallelism, compose TP with PP or FSDP — don't extend TP itself.
- **TP inside attention requires n_heads divisible by TP.** With GQA, the number of KV heads may be small (8 in Llama 3). TP=8 → 1 KV head per rank; TP=16 would split a single head, which Megatron-SP / newer kernels handle with sequence-sharding the head.
- **No overlap in vanilla Megatron.** The four all-reduces per layer sit on the critical path. Recent work (e.g., TP-Aware Dispatch, Mesh-TP overlap) partially overlaps comm with compute by launching the next layer's matmul before the all-reduce completes. Not in vanilla Megatron.
- **All-reduce precision matters.** Gradient all-reduces at FP16 can lose precision at large TP sizes. Llama 3 uses FP32 reduce-scatter for gradients. Megatron-FSDP and Megatron-2 support FP32 collective communication for gradients.
- **Dropout RNG is tricky.** Dropout inside the TP region must use synchronized RNG so all ranks drop the same elements — otherwise the all-reduce would sum mismatched masks. Standard practice: seed dropout from `rank // tp_world_size`.
- **Embedding layer is its own thing.** The input embedding is replicated (not TP-sharded) by default; the LM head is usually TP-sharded along the vocab dimension. LM-head all-reduce is computationally expensive for large vocabularies (128k for Llama 3).
- **Activation checkpointing + TP.** TP-sharded activations are smaller by factor t, so checkpointing is less critical. But composing both works fine.
- **TP=2 is often near-free.** On dual-GPU nodes with NVLink, TP=2 halves per-GPU activation memory with minimal bandwidth cost.
- **Loss computation.** The final CE loss must gather logits from all TP ranks (they each have a vocab-sharded slice). Use a TP-aware CE loss (e.g., Megatron's `vocab_parallel_cross_entropy`).
- **Megatron's configuration knobs.** Key flags: `--tensor-model-parallel-size`, `--sequence-parallel` (enables SP), `--recompute-granularity selective`.

---

## Sources

- Paper: *Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism* — Shoeybi et al., 2019, arXiv 1909.08053 — introduces TP for attention + MLP with f/g operators, 2 all-reduces/layer.
- Paper: *Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM* — Narayanan et al., 2021, arXiv 2104.04473 — TP + PP + DP composition (Megatron-2).
- Paper: *Reducing Activation Recomputation in Large Transformer Models* — Korthikanti et al., 2022, arXiv 2205.05198 — sequence parallelism extension.
- Paper: *The Llama 3 Herd of Models* — Meta, 2024, arXiv 2407.21783 — TP=8 within NVLink, composed with CP/PP/FSDP.
- Code: NVIDIA Megatron-LM — https://github.com/NVIDIA/Megatron-LM.
