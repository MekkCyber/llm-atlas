# Sequence Parallelism (Megatron-SP)

*Depth — shard the LayerNorm/Dropout activations along the sequence dimension within a TP group, so the non-TP-sharded activations also split N_t-ways.*

**TL;DR:** Tensor parallelism shards weights and activations **inside** the attention and MLP blocks — but leaves LayerNorm, Dropout, and the residual stream **replicated** across the TP group. At long context, these replicated activations dominate memory. Megatron-SP shards them along the **sequence dimension** within the same TP group. Converters **g** and **ḡ** transition between sequence-sharded and TP-sharded regimes. Communication: replaces TP's all-reduces with all-gather + reduce-scatter — **same bandwidth cost**, different pattern, much less memory. Distinct from context parallelism (which has its own parallel group).

**Prereqs:** [tensor-parallelism](tensor-parallelism.md), [_communication-primitives](../systems/_communication-primitives.md)
**Related:** [context-parallelism](context-parallelism.md) · [_parallelism](_parallelism.md)

---

## What it is

A refinement of [tensor parallelism](tensor-parallelism.md) from Korthikanti 2022, "Reducing Activation Recomputation in Large Transformer Models."

TP shards:
- **Inside** attention: QKV projections (column-parallel), output projection (row-parallel).
- **Inside** MLP: first linear (column-parallel), second linear (row-parallel).

TP does NOT shard:
- **Outside** attention/MLP: LayerNorm's activations, Dropout masks, residual-stream tensors. These are **replicated** across the TP group.

At short context (S=2k), the non-TP-sharded activations (the "residual stream" of shape `[B, S, H]`) are negligible. At long context (S=128k), they dominate — `10 · B · S · H` bytes per layer per TP rank (Korthikanti's formula), vs `24 · B · S · H / t` for the TP-sharded internal activations.

Megatron-SP shards the outside-the-block activations along the **sequence dimension** within the same TP group. Now every activation in the model is sharded N_t-ways.

### The crucial distinction

**Megatron-SP is NOT the same as context parallelism.** Both shard along the sequence dimension, but:

| Aspect | Megatron-SP | Context Parallelism |
|---|---|---|
| Parallel group | Same as TP group | Separate parallel group |
| What's sharded | LN + Dropout activations (only) | Full Q, K, V (attention dim sharded) |
| Attention compute | Still TP-style (all heads, gather to full S first) | Sequence-distributed (each rank sees local Q) |
| When used | Combined with TP, S ≤ typical training lengths | At very long context (32k+), needs its own parallel dimension |

Think of Megatron-SP as a **completion of TP** — filling in the memory gap that TP leaves. CP is an **extension** — a fundamentally different parallelism axis.

---

## How it works

### The two regimes

A Transformer layer has two types of activation regions:
1. **TP-sharded**: inside attention (after QKV split by head) and inside MLP (after column-parallel gemm).
2. **Sequence-sharded (new with SP)**: LayerNorm input/output, Dropout output, residual stream.

In the sequence-sharded regime, each rank holds `B · S / t · H` instead of `B · S · H`.

### Converters g and ḡ

Transitioning between regimes requires collective operations. Megatron-SP introduces two operators:

- **g** (SP → TP): **all-gather** in forward (gather sharded seq back to full seq on each rank, for TP-style compute); **reduce-scatter** in backward.
- **ḡ** (TP → SP): **reduce-scatter** in forward (distribute TP partial sums back to seq-sharded); **all-gather** in backward.

g and ḡ are conjugate: what g does in forward, ḡ does in backward, and vice versa.

```python
class g(torch.autograd.Function):
    """Transitions SP -> TP. Forward: all-gather seq-sharded input. Backward: reduce-scatter."""
    @staticmethod
    def forward(ctx, x_sp):
        return all_gather(x_sp, dim=1, group=tp_group)  # [B, S/t, H] -> [B, S, H]
    @staticmethod
    def backward(ctx, grad_tp):
        return reduce_scatter(grad_tp, dim=1, group=tp_group)  # [B, S, H] -> [B, S/t, H]

class g_bar(torch.autograd.Function):
    """Transitions TP -> SP. Forward: reduce-scatter TP-reduced output. Backward: all-gather."""
    @staticmethod
    def forward(ctx, x_tp):
        return reduce_scatter(x_tp, dim=1, group=tp_group)
    @staticmethod
    def backward(ctx, grad_sp):
        return all_gather(grad_sp, dim=1, group=tp_group)
```

### The layer

Putting it together, a TP+SP Transformer layer looks like:

```python
def tp_sp_transformer_layer(x_sp):  # x_sp is seq-sharded: [B, S/t, H]
    # Residual 1: LayerNorm in SP regime
    y_sp = layer_norm(x_sp)  # SP; LN parameters replicated across TP group

    # SP -> TP (g: all-gather in forward)
    y_tp = g(y_sp)           # [B, S/t, H] -> [B, S, H]

    # Attention (TP-sharded)
    attn = tp_attention(y_tp)  # [B, S, H], row-parallel produces partial sums

    # TP -> SP (g_bar: reduce-scatter in forward)
    attn_sp = g_bar(attn)    # [B, S, H] -> [B, S/t, H], reduce-scattered

    # Dropout in SP regime
    attn_sp = dropout(attn_sp)

    # Residual add in SP regime
    x_sp = x_sp + attn_sp

    # ... same pattern for MLP ...
    y_sp = layer_norm(x_sp)
    y_tp = g(y_sp)
    mlp_out = tp_mlp(y_tp)
    mlp_sp = g_bar(mlp_out)
    mlp_sp = dropout(mlp_sp)
    x_sp = x_sp + mlp_sp

    return x_sp
```

### The bandwidth trick

A key observation: **ring all-reduce = reduce-scatter + all-gather**. TP's one all-reduce per block is implemented as exactly these two steps:

```
all-reduce(x) = reduce-scatter(x) → all-gather(result)
```

Megatron-SP splits this into its two halves — `reduce-scatter` becomes ḡ's forward, `all-gather` becomes g's forward on the **next** block. Net: same total bandwidth, same number of bytes moved.

Communication per layer (fwd+bwd):
- TP alone: 4 all-reduces → effectively `4 · (reduce-scatter + all-gather)` = 8 collective halves.
- TP + SP: 4 all-gathers (g fwd + ḡ bwd × 2) + 4 reduce-scatters (ḡ fwd + g bwd × 2) = 8 collective halves.

**Identical bandwidth.** Just different partitioning of the same bytes across the same operations.

### Memory effect — Eq. 4 of Korthikanti 2022

Activation memory per Transformer layer per GPU:

**TP alone**:
```
activations_TP_per_layer = s · b · h · (10 + 24/t + 5 · a · s / (h · t))
```

- The `10` term = LN + Dropout + residual, replicated across TP.
- The `24/t` = inside-attention + inside-MLP, TP-sharded.
- The `5·a·s/(h·t)` = attention matrix term (a = num heads).

**TP + SP**:
```
activations_TP_SP_per_layer = (s · b · h / t) · (34 + 5 · a · s / h)
```

- Everything is now divided by `t`. The `10` term → `10/t`. The `24` → `24/t`. The attention matrix term → also /t.

Net: activation memory per GPU **drops by ~t×** when SP is added.

### Combined with selective recomputation

Korthikanti's paper pairs SP with **selective activation recomputation**: recompute only the activations inside attention (which are cheap to recompute) and keep the rest. Together:

- Activation memory reduced by **5×** vs vanilla TP.
- Recomputation time overhead reduced by **>90%** vs full recomputation.

Concrete example from the paper: 530B GPT-3-style model on 2240 A100s reaches **54.2% MFU** with SP + selective recompute, vs 42.1% with vanilla TP + full recompute.

---

## Why it matters

- **Zero-overhead memory win.** Same bandwidth as TP, strictly smaller activation memory. Pure improvement over TP.
- **Enables larger micro-batches and longer context within TP.** The memory savings often let you drop activation checkpointing, speeding up training.
- **Composes trivially with TP.** SP is not a separate parallelism dimension; it's a TP optimization. Same parallel group.
- **Part of modern production stacks.** Megatron-LM enables SP by default via `--sequence-parallel`. Most frontier dense runs use TP + SP.

---

## Gotchas & tricks

- **Always pair with TP.** SP alone doesn't make sense — it uses the TP group as its sharding axis.
- **S must be divisible by t.** The sequence dimension is split into t chunks; the boundary behavior is undefined otherwise.
- **Dropout RNG.** Dropout must use **different** RNG state across SP ranks (because each rank has a different chunk of the sequence), but **same** RNG within a single rank across micro-batches. Megatron's `tensor_parallel.checkpoint.get_cuda_rng_tracker()` handles this.
- **LayerNorm parameters.** LN scale/shift parameters are **replicated** across TP; their gradients are computed redundantly by all ranks → need to average via all-reduce in backward. Megatron handles this.
- **Don't confuse with CP.** CP is a separate parallel axis with its own group. SP shares the TP group.
- **Limited effective scale.** SP scales only to TP's ceiling (~8). For longer context you still need CP on top.
- **FlashAttention integration.** SP works fine with FA2/FA3 — attention compute runs on the gathered full-S representation inside the TP region.
- **BF16 vs FP32 for collective precision.** Reduce-scatter in BF16 can lose precision at large t. Modern implementations keep gradient reduce-scatter in FP32.
- **Debugging.** If you see activation-memory numbers that don't scale with t, SP is probably broken (likely missing converter). Compare expected `(10 + 24)/t · BSH` with actual.
- **Compatibility with PP.** SP is applied within each PP stage. Works fine.

---

## Sources

- Paper: *Reducing Activation Recomputation in Large Transformer Models* — Korthikanti et al., 2022, arXiv 2205.05198 — the Megatron-SP paper, §4.1 introduces the g/ḡ operators and the memory math.
- Paper: *Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism* — Shoeybi et al., 2019, arXiv 1909.08053 — the TP baseline.
- Code: NVIDIA Megatron-LM — https://github.com/NVIDIA/Megatron-LM — `--sequence-parallel` flag.
