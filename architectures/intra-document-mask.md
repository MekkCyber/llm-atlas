# Intra-Document Attention Masking
*Depth — block self-attention across packed-document boundaries in a training sequence.*

**TL;DR:** To maximize GPU utilization, training sequences are **packed**: multiple independent documents concatenated into a single `S`-token sequence with separators between them. Vanilla attention lets every token attend to every earlier token in the sequence — including tokens from different documents. **Intra-document masking** adds an attention mask that **prevents attention across document boundaries**. Small effect at short context (attention rarely reaches far); important at long context (128k packing may have dozens of documents, and the model should not mix their content). Used throughout Llama 3's pretraining (Sec. 3.2).

**Prereqs:** [attention](../fundamentals/attention.md), [transformer-block](transformer-block.md)
**Related:** [gqa](gqa.md) · [context-parallelism](../pre-training/context-parallelism.md)

---

## What it is

### Packed sequences

Pretraining data is documents of varying lengths — some 100 tokens, some 10k. Training at a fixed sequence length (say 8192) naively would require **padding** short documents, wasting compute on padded positions. The standard alternative: **pack multiple documents** into a single training sequence.

```
[Doc 1: 3200 tokens] [SEP] [Doc 2: 1800 tokens] [SEP] [Doc 3: 2800 tokens] [SEP] [Doc 4: ...]
|-----------------------------  8192 tokens  -----------------------------|
```

Each document ends with a separator token (e.g., `<|endoftext|>` or `<|eot_id|>`). Training loss is computed over every position, but loss on the separator token is usually ignored.

### The problem without masking

Vanilla causal attention lets every position `i` attend to every position `j < i`. In a packed sequence, position 5000 (inside Doc 2) can attend to positions 0–3199 (all of Doc 1).

Empirically:
- **Short context (2k, 4k)**: attention rarely "reaches back" far enough to cross document boundaries with any significant weight. The model tends to focus on nearby tokens. Inter-document attention is weak noise.
- **Long context (32k, 128k)**: attention *can* reach arbitrarily far; the model may learn to attend across document boundaries, mixing information that should be independent. This hurts long-context understanding — the model sees concatenated documents as one weird long document.

### The fix

Modify the attention mask so that position `i` can only attend to positions `j < i` that belong to the **same document**.

```
Mask[i, j] = 1 (allowed) if:
    j < i (causal)           AND
    document(i) == document(j)
```

Otherwise `Mask[i, j] = 0` (blocked).

---

## How it works

### Building the mask

You need a "document ID" per position — which document does this position belong to?

```python
def build_intra_document_mask(doc_ids):
    # doc_ids: [B, S] tensor of integer document IDs
    # Return: [B, 1, S, S] additive mask (0 or -inf)
    S = doc_ids.size(-1)
    causal = torch.tril(torch.ones(S, S, dtype=torch.bool))                    # [S, S]
    same_doc = doc_ids.unsqueeze(-1) == doc_ids.unsqueeze(-2)                  # [B, S, S]
    allowed = causal.unsqueeze(0) & same_doc                                    # [B, S, S]

    mask = torch.zeros(B, 1, S, S)
    mask.masked_fill_(~allowed.unsqueeze(1), float('-inf'))
    return mask

# Apply in attention:
attn_scores = Q @ K.transpose(-2, -1) / sqrt(d_head)
attn_scores = attn_scores + mask           # -inf positions become 0 after softmax
attn = softmax(attn_scores)
out = attn @ V
```

### Variable-length (varlen) attention

Instead of materializing a `B × S × S` mask, modern attention kernels (Flash Attention 2+) support **variable-length attention**. You pack the B×S tokens into a single flattened tensor plus a `cu_seqlens` array of cumulative document lengths:

```
tokens        = [t0, t1, t2, ...]                   # packed, no batch dim
cu_seqlens    = [0, 3200, 5000, 7800, 8192]         # document boundaries
```

Flash Attention's `flash_attn_varlen_func` handles this natively, internally computing block-wise attention within each `[cu_seqlens[i], cu_seqlens[i+1])` segment. **Zero extra memory cost** — the document mask is implicit in the block structure.

This is the production implementation. No explicit `[B, S, S]` mask is ever materialized.

### Interaction with document separators

The separator token between documents:
- **Its attention in**: under intra-document masking, nothing after the separator can attend to it (from a different doc's perspective) because the separator's document ID is typically the preceding document's ID.
- **Its loss**: usually masked out (no loss computed on separator tokens) — you don't want the model to spend capacity predicting separators.

### Llama 3's choice

Llama 3 enables intra-document masking throughout pretraining (Sec. 3.2: *"This change had limited impact during standard pre-training, but it was important in continued pre-training on very long sequences."*). Specifically:

- Short-context phase (4k, 8k): minimal effect.
- **Long-context extension phase (32k, 128k)**: important — without it, the model learns bad cross-document reasoning patterns.

### Interaction with CP

Document masks cause trouble for [Ring Attention](../pre-training/context-parallelism.md)'s ring-based context parallelism, because different K/V blocks may need different mask patterns. Llama 3's all-gather CP avoids this — once full K/V is materialized, any mask works cleanly.

---

## Why it matters

- **Enables long-context training without quadratic-attention pollution.** At 128k, packed sequences have dozens of documents; without masking, they'd all cross-contaminate.
- **Cheap.** Varlen attention is essentially free vs regular causal attention in modern kernels. No compute overhead.
- **Prerequisite for packed-document training at long context.** If you can't pack (wasting compute) or can't mask (training on contaminated sequences), long-context training becomes much more expensive.
- **Compatibility with RL / SFT**. In SFT/RLHF, one "document" = one conversation. Intra-document masking ensures the model doesn't attend across conversations in packed training batches.

---

## Gotchas & tricks

- **Effect size is context-length dependent.** At 2k context, intra-document masking barely moves metrics. At 128k, it's essential. Most benchmarks at 8k won't show a delta — don't over-extrapolate from small-context ablations.
- **Document separator placement.** Some recipes use a separator token (`<|endoftext|>`), others just split by document boundaries without a token. Either works for masking purposes, but the tokenizer must be consistent.
- **Loss masking on separators.** Usually skip loss on separator tokens to avoid wasting capacity on "predict the next separator."
- **Positional encoding resets.** With intra-document masking, should each document start at position 0, or continue the running position? Llama 3 does *not* reset positions — positions are continuous across the packed sequence. RoPE just sees the absolute position within the packed window. This works because attention is blocked across documents; positional info doesn't leak.
- **With RoPE, same-document attention still uses relative positions within the document.** Position 3200 attending to position 3000 within the same document sees a RoPE rotation of angle `(3200 − 3000) · θ` = same as if they were at positions 200 and 0 of an unpacked single document (modulo RoPE's linearity).
- **Backward compatibility.** A model trained WITH intra-document masking should be INFERENCED with a "clean" sequence (no packing). The masking is only about training; at inference, each conversation / document is its own context.
- **Don't use for chat.** Multi-turn chat is one coherent document; don't mask across turns.
- **Flash Attention varlen is the preferred implementation.** `flash_attn_varlen_func` in FA2+ handles this natively with `cu_seqlens`. Avoid hand-rolled mask tensors.
- **Composes with sliding-window attention.** If your model uses sliding window (Mistral 7B), intra-document masking composes — the mask is `causal AND same_doc AND within_window`.

---

## Sources

- Paper: *The Llama 3 Herd of Models* — Meta, 2024, arXiv 2407.21783, Sec. 3.2 — explicit use of intra-document attention masking.
- Paper: *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning* — Dao, 2023, arXiv 2307.08691 — varlen attention for packed sequences.
- Paper: *Efficient Training of Language Models to Fill in the Middle* — Bavarian et al., 2022, arXiv 2207.14255 — earlier discussion of document packing for efficient training.
- Code: Flash Attention — https://github.com/Dao-AILab/flash-attention — `flash_attn_varlen_func`.
