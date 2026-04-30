# Grouped-Query Attention (GQA)
*Depth — share K and V projections across groups of query heads to shrink KV cache and speed decoding.*

**TL;DR:** Standard multi-head attention (MHA) has one Q projection and one K/V projection **per head**. With 128 heads and head dim 128, 128 KV projections blow up the KV cache at inference. **GQA** shares each K/V projection across a **group** of query heads: 128 query heads, 8 KV heads → 16 query heads share each KV head. Halves or quarters KV-cache memory at near-zero accuracy cost. Llama 2 70B, Llama 3 all sizes, and most modern LLMs use GQA with 8 KV heads. The middle ground between MHA (full quality, expensive cache) and MQA (1 KV head shared across all queries; cheap cache, quality drop).

**Prereqs:** [attention](../fundamentals/attention.md), [multi-head-attention](multi-head-attention.md)
**Related:** [mla](mla.md) · [transformer-block](transformer-block.md)

---

## What it is

Three attention variants, distinguished by how many distinct K/V projections the model has:

| Variant | Q heads | KV heads | KV cache |
|---|---|---|---|
| **MHA** (vanilla) | H | H | Full |
| **GQA** (g groups) | H | H/g | H/g × smaller |
| **MQA** (one KV) | H | 1 | H× smaller |

Standard MHA: every query head has its own K and V projection. For H heads of dim d, K and V each have dimension H·d — same as Q.

Multi-Query Attention (MQA, Shazeer 2019): *all* query heads share a **single** K and V projection. K and V each have dimension d (not H·d). KV cache shrinks by H×.

Grouped-Query Attention (GQA, Ainslie 2023): divide H query heads into g groups; each group shares **one** K and V projection. K, V each have dimension g·d.

- g = H → MHA.
- g = 1 → MQA.
- g in between → GQA.

Llama 3 uses **g = 8** regardless of model size (8B, 70B, 405B). 8B has 32 Q heads (4 per group); 70B has 64 Q heads (8 per group); 405B has 128 Q heads (16 per group).

---

## How it works

### The math

Standard MHA forward (per head):

```
Q_h = X · W_Q_h       shape [B, S, d]
K_h = X · W_K_h       shape [B, S, d]
V_h = X · W_V_h       shape [B, S, d]
out_h = softmax(Q_h · K_h^T / √d) · V_h
```

GQA with g groups:

```
Q_h = X · W_Q_h       for h in 0..H-1         shape [B, S, d]  (full H heads)
K_j = X · W_K_j       for j in 0..g-1         shape [B, S, d]  (only g)
V_j = X · W_V_j       for j in 0..g-1         shape [B, S, d]  (only g)

# Each query head h uses K, V from group j = h // (H/g)
out_h = softmax(Q_h · K_{h//(H/g)}^T / √d) · V_{h//(H/g)}
```

In practice: you compute the g K/V projections, then "repeat" (broadcast) each K, V across the H/g query heads of its group:

```python
def gqa_forward(x, W_Q, W_K, W_V, W_O, H, g):
    # x: [B, S, H*d]; H = total heads, g = number of KV groups
    d = x.shape[-1] // H
    H_per_group = H // g

    Q = (x @ W_Q).view(B, S, H, d).transpose(1, 2)           # [B, H, S, d]
    K = (x @ W_K).view(B, S, g, d).transpose(1, 2)           # [B, g, S, d]
    V = (x @ W_V).view(B, S, g, d).transpose(1, 2)           # [B, g, S, d]

    # Expand K, V to match Q's head count
    K = K.repeat_interleave(H_per_group, dim=1)              # [B, H, S, d]
    V = V.repeat_interleave(H_per_group, dim=1)              # [B, H, S, d]

    attn = softmax(Q @ K.transpose(-2, -1) / sqrt(d))        # [B, H, S, S]
    out = attn @ V                                            # [B, H, S, d]
    out = out.transpose(1, 2).reshape(B, S, H * d)
    out = out @ W_O
    return out
```

In production, the repeat_interleave is often skipped and replaced with a broadcasted matmul to save memory. Flash Attention 2+ has native GQA support — you just pass differently-sized K/V tensors.

### Parameter count

For model hidden H_model, total heads H, head dim d = H_model/H, KV groups g:

- `W_Q`: H_model × H_model (same as MHA).
- `W_K`: H_model × g·d (g/H × smaller than MHA).
- `W_V`: H_model × g·d (g/H × smaller than MHA).
- `W_O`: H_model × H_model (same as MHA).

For Llama 3 70B (H_model=8192, H=64, d=128, g=8): W_K and W_V are 8192 × 1024 instead of 8192 × 8192 — **8× smaller**. Savings across 80 layers ≈ 800M parameters.

### KV cache at inference

At inference, you autoregressively decode; the KV cache stores K and V for every position in the context. Memory per layer per token:

- MHA: `2 · H · d = 2 · H_model` bytes per token per layer (in BF16, 2 bytes/elem).
- GQA: `2 · g · d = 2 · g · d` bytes per token per layer — **H/g× smaller**.

For Llama 3 70B at 32k context, batch 1:
- MHA: 2 · 8192 · 32768 · 80 = ~40 GB of KV cache.
- GQA (g=8): 2 · 1024 · 32768 · 80 = ~5 GB.

This is the decisive win: GQA makes long-context inference memory-practical. MHA at 128k context would require a dedicated box per active user.

### Decoding throughput

Decode is bandwidth-bound on the KV cache. Per decode step, every layer reads the full KV cache to compute attention. Throughput scales inversely with KV-cache size:

- Larger KV cache → more bytes to read per step → slower decode (bandwidth-bound).
- GQA's 8× smaller cache → ~8× higher decode throughput at long context (within the bandwidth-bound regime).

This matters for serving: GQA-based models serve at much higher tokens/sec/GPU than MHA equivalents.

---

## Training

### Uptraining from MHA

The Ainslie 2023 paper introduces an "uptraining" procedure: take an MHA-pretrained model and **convert** it to GQA. Mean-pool the K (and V) projections within each group to initialize the GQA K/V. Fine-tune for **5% of the original pretraining compute** — recovers most of the MHA quality.

Practical use: you don't have to commit to GQA at pretraining time; you can convert later. But new frontier models just use GQA from scratch.

### Quality vs MHA/MQA

Ainslie 2023's main result (Table 1, arXiv 2305.13245):

| Variant | Quality | KV cache |
|---|---|---|
| MHA (baseline) | 100% (reference) | 1× |
| GQA-8 | 99.5% | 1/8× |
| GQA-1 = MQA | 95% | 1/H× |

GQA with 8 groups recovers **essentially all** of MHA's quality while paying only 1/8th the KV-cache cost. MQA (g=1) loses a measurable amount of quality.

The "8 KV heads" choice has become near-universal — Llama 2 70B, all Llama 3, Qwen2, Gemma 2, DeepSeek (partially; they use MLA), Mistral. It's one of the most consistent architectural choices across frontier models.

---

## Why it matters

- **KV cache is the inference bottleneck.** For long-context serving, KV cache dominates GPU memory. Without GQA, 128k context is unaffordable at scale.
- **Essentially free quality.** GQA-8 gives up <1% quality vs MHA. For the 8× cache reduction, this is an obvious trade.
- **Pairs with long-context training.** Llama 3's 128k context extension would be much harder without GQA — both training (less activation memory) and inference (less KV cache) benefit.
- **Ubiquity.** Almost every frontier LLM since Llama 2 70B uses GQA or a further evolution (MLA in DeepSeek).

---

## Gotchas & tricks

- **Group size 8 is the default.** It's not special — just what Ainslie tested and what Llama propagated. Some MoE models use g=4 or g=16. The quality/cache trade-off plateaus around g=8.
- **Head dim d still matters.** Total KV cache scales with `g · d`. Most modern models keep d=128 and vary g. Increasing d instead of g is an unexplored knob.
- **GQA + TP interacts.** Tensor parallelism shards along the head dim. With g=8 and TP=8, each rank gets 1 KV head. TP > g forces splitting a single KV head — Megatron-LM + FlashAttention handles this with "split-k" kernels.
- **Don't confuse with MLA.** DeepSeek-V2/V3's [MLA](mla.md) is a different approach — compresses KV to a low-rank latent, decompresses on demand. MLA achieves a similar goal (shrink KV cache) with different machinery; it's not GQA.
- **RoPE integrates.** RoPE is applied per head, to both Q and K. With GQA, RoPE is applied to each of the g K heads (not the broadcasted copies). Q is still rotated per-query-head.
- **Position encoding precision.** At long context, per-head RoPE tables should be FP32. See [rope](../fundamentals/rope.md).
- **Flash Attention native support.** FA2+ has a `num_groups` / `num_heads_k` parameter that takes the smaller KV head count. No repeat_interleave needed.
- **vLLM and other serving frameworks.** GQA is first-class in vLLM's paged attention — the KV cache is allocated at g-head granularity, not H-head.
- **Context extension via RoPE base works unchanged.** GQA doesn't interfere with long-context extension (YaRN, ABF).
- **Distillation from MHA to GQA.** An MHA teacher's outputs can distill into a GQA student with no modification to the distillation loss. Common pattern for cheap conversion.

---

## Sources

- Paper: *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints* — Ainslie et al., 2023, arXiv 2305.13245 — introduces GQA, including the uptraining recipe.
- Paper: *Fast Transformer Decoding: One Write-Head is All You Need* — Shazeer, 2019, arXiv 1911.02150 — MQA, the extreme form.
- Paper: *Attention Is All You Need* — Vaswani et al., 2017 — MHA, the baseline.
- Paper: *The Llama 3 Herd of Models* — Meta, 2024, arXiv 2407.21783 — uses GQA with 8 KV heads across all sizes.
- Paper: *Llama 2* — Touvron et al., 2023, arXiv 2307.09288 — first large-scale use of GQA at 70B.
