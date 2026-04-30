# Context Parallelism (CP)

*Depth — shard the sequence dimension across GPUs so 128k+ context attention fits in memory.*

**TL;DR:** Attention is O(S²) in memory. At S = 128k, the attention matrix alone is 16 GB per head. Context parallelism (CP) splits the sequence into chunks across N_cp GPUs: each rank holds only `S / N_cp` tokens. Two main variants: **Ring Attention** (Liu 2023) rotates K/V blocks around a ring, overlapping comm with compute; **all-gather CP** (Llama 3) all-gathers K/V once per layer and computes local Q-attention. Ring scales further but has load-balance / mask complications; all-gather is simpler and wins when K/V are small (GQA/MQA).

**Prereqs:** [_parallelism](_parallelism.md), [_communication-primitives](../systems/_communication-primitives.md), [attention](../fundamentals/attention.md)
**Related:** [fsdp](fsdp.md) · [tensor-parallelism](tensor-parallelism.md) · [sequence-parallelism](sequence-parallelism.md)

---

## What it is

In attention `out = softmax(Q K^T / √d) V`:
- Q, K, V are `[B, n_heads, S, d_head]` — activation memory is `O(B · S · H)`.
- The attention matrix is `[B, n_heads, S, S]` — the explicit form is `O(B · S²)`, which is the dominant memory term at long context.

TP shards across heads. FSDP shards across DP. Neither shards along the sequence dimension.

CP shards along the **sequence dimension**. Each rank holds `S / N_cp` tokens — one sequence chunk per rank. The challenge: computing attention requires Q-tokens to interact with **all** K-tokens, but each rank only has a slice of K. Two strategies to resolve this:

1. **Ring Attention** (Liu 2023): rotate K/V chunks around the ring of CP ranks, accumulating attention output incrementally via online softmax.
2. **All-gather CP** (Llama 3): all-gather K/V once before attention, compute local Q vs full K/V.

---

## Variant 1: Ring Attention (Liu 2023)

### Algorithm

Given N_cp ranks, each holding local chunk `Q_i, K_i, V_i` (of length `S / N_cp`):

```
# N_cp iterations around the ring
for iter in range(N_cp):
    send K_i, V_i to rank (i+1) % N_cp   (async)
    recv K_j, V_j from rank (i-1) % N_cp  (async)
    # meanwhile, compute partial attention on local Q_i with the K, V block we currently have
    partial = flash_attention_blockwise(Q_i, K_current, V_current)
    # accumulate into output via online softmax (carry running max + running denominator + output)
    update(out_i, partial)
    K_current, V_current = K_j, V_j
```

After N_cp iterations, every Q chunk has seen every K/V chunk — so we've computed the full attention exactly.

### Online softmax

A key technical detail: online softmax (Milakov & Gimelshein 2018, used in Flash Attention) lets us accumulate `softmax(Q K^T / √d) V` **incrementally**, chunk-by-chunk along K, without materializing the full `S × S` matrix. At each step we maintain:

- `m_i`: running max of Q K^T so far.
- `l_i`: running denominator (sum of `exp(qk - m_i)` so far).
- `o_i`: running output numerator (sum of `exp(qk - m_i) · V` so far).

When a new K/V chunk arrives:
```python
def update(m, l, o, K_new, V_new, Q):
    qk = Q @ K_new.T / sqrt(d)                  # new partial scores
    m_new = max(m, qk.max(dim=-1))              # new running max
    l_rescale = exp(m - m_new)                  # rescale old sums
    o_rescale = o * l_rescale                   # rescale old numerator
    l_rescale *= l                              # rescale old denominator
    exp_qk = exp(qk - m_new)                    # new block's contribution
    o_new = o_rescale + exp_qk @ V_new
    l_new = l_rescale + exp_qk.sum(dim=-1)
    return m_new, l_new, o_new

# Final: out = o / l
```

This accumulator is numerically stable and mathematically exact. Ring Attention uses exactly this to fold in each K/V chunk as it arrives.

### Memory

Per rank: `O(S / N_cp)` activation memory. Scales linearly with N_cp — 128 ranks → 128× the effective context length.

### Communication

Per iteration: send `B · (S/N_cp) · H_kv` bytes (K chunk) + same for V. Total N_cp iterations → `B · S · H_kv` bytes transferred per rank per layer.

Liu 2023's key claim: **as long as block compute takes longer than block transfer, communication is fully hidden**. This is true for long contexts where per-chunk attention compute is substantial.

### The causal-mask problem

With causal attention, later query chunks need to attend to *more* K/V chunks (all preceding positions), while early query chunks need *fewer*. Naïve assignment — rank i holds tokens `i · (S/N_cp)` to `(i+1) · (S/N_cp)` — gives rank N_cp-1 maximal work and rank 0 minimal work, with **N_cp×** load imbalance.

**Fix**: **zig-zag / striped sharding**. Each rank holds **two** chunks: the i-th and the (2N_cp − 1 − i)-th. Now each rank has roughly equal total work. Llama 3 adopts this pattern even in its all-gather variant (see below).

### Arbitrary masks

Causal is the common case. For **document masks** (packed sequences with no-cross-document attention) or sliding-window masks, Ring Attention must track which K/V blocks are actually needed by which Q blocks — the uniform ring schedule may do wasted work or require extra control flow.

---

## Variant 2: All-gather CP (Llama 3)

### The trick

Instead of rotating K/V around the ring, **all-gather K and V** before attention. Each rank then has the full K, V and computes attention on its local Q chunk.

```python
# Per rank
Q_local = local query chunk (S/N_cp tokens)
K_local = local key chunk   (S/N_cp tokens)
V_local = local value chunk (S/N_cp tokens)

K_full = all_gather(K_local, group=cp_group)   # [B, n_kv_heads, S, d_head]
V_full = all_gather(V_local, group=cp_group)

out_local = flash_attention(Q_local, K_full, V_full)  # local Q vs global K, V
```

### Why Llama 3 picked this

**Under GQA**, K and V have far fewer heads than Q:
- Llama 3: n_Q_heads = 128 (405B), n_KV_heads = 8. So K/V are **16× smaller** than Q in memory.
- All-gather cost scales with K/V size — with GQA it's effectively 16× cheaper than it would be for a full-MHA model.

Plus:
- **Supports arbitrary masks.** Once full K/V is materialized, any attention mask works — document mask, sliding window, whatever. No ring-schedule bookkeeping.
- **Simpler implementation.** One collective per layer vs N_cp-step ring loop.
- **Works with any attention kernel.** Flash Attention 2 / 3 work out of the box on the full-K local-Q attention.

Trade-off: memory for full K/V is `O(S)` per rank (not `O(S/N_cp)`). For K, V this is still small under GQA; for Q the local chunk stays sharded.

### The sharding

Llama 3's CP uses the **zig-zag partition** (like Ring Attention's causal load-balancing): sequence is split into `2 · N_cp` chunks, each rank gets chunks `i` and `2·N_cp − 1 − i`. Balances work across ranks.

### Communication cost

One all-gather per attention layer for K, one for V:
- Per rank: `B · S · H_KV · 2` bytes per layer (K and V combined).
- Over L layers: `2 · L · B · S · H_KV` bytes per rank per step.

For Llama 3 405B (L=126, H_KV = 8 × 128 = 1024) at S = 128k, B = 1: `2 · 126 · 128k · 1024 · 2 = 66 GB` per step per rank. At NVLink+RoCE rates, comparable to other comm terms — not dominant.

### When all-gather beats ring

- **GQA / MQA**: K/V small → cheap all-gather.
- **Arbitrary masks**: document masks, sliding windows, etc.
- **Short-ish long context** (32k–128k, not 1M+): memory for full K/V per rank is manageable.

### When ring beats all-gather

- **Full MHA**: K/V same size as Q → all-gather expensive.
- **Extreme context** (>1M): even K/V too big to materialize per rank.
- **When the ring's compute-comm overlap is favorable.**

---

## How it works — detailed forward

### Ring Attention forward

```python
def ring_attention(Q_local, K_local, V_local, cp_group):
    cp_size = cp_group.size()
    cp_rank = cp_group.rank()

    # Accumulator state for online softmax
    m = -inf * torch.ones(B, n_heads, S_local)          # running max
    l = torch.zeros(B, n_heads, S_local)                # running denominator
    o = torch.zeros(B, n_heads, S_local, d_head)        # running numerator

    K_cur, V_cur = K_local, V_local

    for step in range(cp_size):
        # Async start send/recv for next iteration (if not last)
        if step < cp_size - 1:
            recv_req = irecv(K_next, src=(cp_rank - 1) % cp_size)
            send_req = isend(K_cur,  dst=(cp_rank + 1) % cp_size)
            # similarly for V

        # Compute attention of local Q vs current K, V
        partial_scores = Q_local @ K_cur.transpose(-2, -1) / sqrt(d_head)

        # Apply mask (causal, document, etc.) for this (Q_local, K_cur) block pair
        mask = get_mask_for_blocks(cp_rank, (cp_rank - step) % cp_size)
        partial_scores.masked_fill_(mask, -inf)

        # Online softmax update
        block_max = partial_scores.max(dim=-1, keepdim=True).values
        new_m = torch.maximum(m, block_max)
        l_rescale = torch.exp(m - new_m)
        o = o * l_rescale.unsqueeze(-1)
        l = l * l_rescale
        exp_scores = torch.exp(partial_scores - new_m)
        o = o + exp_scores @ V_cur
        l = l + exp_scores.sum(dim=-1)
        m = new_m

        # Complete async comm
        if step < cp_size - 1:
            recv_req.wait()
            send_req.wait()
            K_cur, V_cur = K_next, V_next

    # Final attention output
    out_local = o / l.unsqueeze(-1)
    return out_local
```

### All-gather CP forward

```python
def all_gather_attention(Q_local, K_local, V_local, cp_group):
    # Collective: gather K, V from all cp ranks
    K_full = all_gather(K_local, group=cp_group)   # [B, n_kv, S, d_head]
    V_full = all_gather(V_local, group=cp_group)   # [B, n_kv, S, d_head]

    # Standard attention: local Q, full K, V
    out_local = flash_attention(Q_local, K_full, V_full, causal=True)
    return out_local
```

### Backward

Ring Attention's backward runs the ring in reverse direction, accumulating Q, K, V gradients via another online computation. Mechanically nontrivial; Flash Attention 3 exposes a ring-CP backward kernel.

All-gather CP's backward is simpler: gather K, V gradients per rank, reduce-scatter them back to the K, V owner (inverse of forward's all-gather).

---

## Why it matters

- **The only way to 128k+ context at scale.** Attention memory is the dominant per-rank cost at long context. CP shards it.
- **Orthogonal to other axes.** Composes with TP, PP, FSDP. Llama 3: 4D `[TP=8, CP=16, PP=16, DP=8]`.
- **Different sweet spots.** Ring attention is the research favorite (cleaner theoretical memory story); all-gather CP is the production favorite for GQA models with arbitrary masks.
- **Flash Attention integration.** Both variants compose with Flash Attention 2 / 3 kernels. Ring needs a custom accumulator; all-gather is a drop-in.

---

## Gotchas & tricks

- **Load balance is the rubber-meets-road problem.** Causal masking → zig-zag sharding is not optional for Ring CP. Also applies to all-gather CP if the work is imbalanced.
- **GQA + CP is the blessed combo.** Small K/V heads → cheap comm. MHA + CP is painful.
- **Document masks work cleanly with all-gather CP, painfully with Ring.** If your training uses packed sequences with document boundaries, prefer all-gather.
- **Inference.** Ring Attention for inference is tricky (KV cache per rank needs special handling). All-gather CP at inference just all-gathers the cache — simpler.
- **Flash Attention + Ring.** FA3 includes `flash_attn_with_kvcache` and ring-friendly variants. Older FA versions don't support Ring cleanly.
- **Communication overlap.** Ring's whole pitch is "comm hidden by compute." At short block lengths the overlap is imperfect — comm spills into the critical path. Block size must be large enough.
- **All-gather CP within NVLink.** Like TP, all-gather CP benefits from NVLink bandwidth. Can extend across NVLink+RoCE but performance depends on K/V size and comm topology.
- **Don't confuse CP with Megatron sequence parallelism.** [Sequence parallelism](sequence-parallelism.md) shards LN/Dropout activations along seq dim within a TP group — it's a TP-companion, not a separate axis. CP shards the full sequence across a dedicated parallelism group and touches attention directly.
- **Ring CP's startup cost.** First N_cp ring iterations dominate small-context runs. CP only pays off once S/N_cp is large enough that block compute > block comm.
- **Numerical precision.** Online softmax accumulators should run in FP32 to avoid catastrophic cancellation across many ring iterations. BF16 accumulators at large N_cp can lose precision.

---

## Sources

- Paper: *Ring Attention with Blockwise Transformers for Near-Infinite Context* — Liu et al., 2023, arXiv 2310.01889 — canonical ring-based CP.
- Paper: *The Llama 3 Herd of Models* — Meta, 2024, arXiv 2407.21783 — all-gather CP with zig-zag sharding.
- Paper: *Striped Attention* — Brandon et al., 2023, arXiv 2311.09431 — load-balanced ring attention for causal masking.
- Paper: *FlashAttention-2* — Dao, 2023, arXiv 2307.08691 — the online-softmax kernel the CP variants are built on.
- Paper: *FlashAttention-3* — Shah et al., 2024, arXiv 2407.08608 — FA3 ring + CP variants.
- Paper: *Online normalizer calculation for softmax* — Milakov & Gimelshein, 2018, arXiv 1805.02867 — the online softmax algorithm.
