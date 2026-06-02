# Sliding-Window Attention (SWA)

*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Restrict self-attention to a fixed-size window of recent tokens — each query at position $t$ attends only to keys at positions $[t - W, t]$ for window size $W$. Cost per layer drops from $O(L^2)$ to $O(LW)$ in compute and $O(W)$ per-token in KV cache. Stacking SWA layers gives an *effective* receptive field of $W \cdot D$ tokens via deep composition. Practical recipes (Mistral, Mellum 2) mix SWA layers with periodic full-attention layers to keep some global mixing.

**Prereqs:** [attention](../fundamentals/attention.md), [multi-head-attention](multi-head-attention.md)
**Related:** [transformer-block](transformer-block.md) · [_moe](_moe.md)

---

## What it is

Standard self-attention costs $O(L^2)$ FLOPs per layer for sequence length $L$ and $O(L)$ KV-cache per token at inference. At long contexts these terms dominate everything else in the model. The simplest fix is locality: most token-to-token dependencies live within a few hundred or few thousand tokens; you don't need every query to look at every key.

Sliding-window attention enforces this. Each query position $t$ attends to a fixed window $[t - W, t]$ of preceding (and including) keys, masking everything else. The attention mask is the standard causal mask **intersected** with a band mask of width $W$.

Compute per layer drops from $O(L^2)$ to $O(L \cdot W)$ — linear in $L$ for fixed $W$. KV cache per token drops from "everything you've seen" to "the last $W$ tokens" for that layer.

---

## How it works

### The mask

```
For query position t:
    valid_keys = [max(0, t - W + 1), t]    # window of size W, causal
    mask out everything else
```

The attention computation is otherwise unchanged — same softmax, same value mixing.

### Deep-composition receptive field

A single SWA layer has receptive field $W$. Two stacked SWA layers have receptive field $2W - 1$ (the second layer's window of windows). With $D$ SWA layers, the effective receptive field is roughly $D \cdot W$. So a model with 32 SWA layers of window 4096 can mix information from ~131K tokens away even though no single layer sees that far.

This is the same "dilated receptive field" intuition as CNNs.

### Interleaving with full attention

Pure-SWA models lose the ability to directly mix anywhere-to-anywhere on a single layer, which hurts tasks requiring long-range pointer lookup (retrieval, code referencing). Mixed recipes interleave:

- **Mistral 7B** (Jiang et al. 2023): every layer is SWA with window 4096.
- **Mellum 2** (2026): 3 of every 4 layers are SWA; the 4th is full attention. The full-attention layers serve as long-range "express links" for information the SWA layers can only relay over many steps.
- **Gemma 2** and others use similar mixed patterns at various ratios.

### Compatibility with YaRN / RoPE

[YaRN](yarn.md) context extension typically interpolates RoPE frequencies across all positions. Under SWA, the windowed layers don't see far enough for the long-range frequencies to matter; YaRN is often applied **layer-selectively** to just the full-attention layers in mixed-SWA models.

---

## Why it matters

- **Long-context cost reduction.** At 128K context, full attention is wildly memory- and compute-bound; SWA brings most layers back into a tractable regime.
- **Inference KV cache.** Memory-bound batch-1 decode is dominated by KV reads. SWA caps the per-layer cache size, slashing decode latency for long contexts.
- **Compatible with GQA, MoE, and most modern attention tricks.** SWA is orthogonal to KV-head sharing and expert mixing — modern recipes stack all three.

---

## Gotchas & tricks

- **Pure SWA loses long-range pointer lookup.** If your task needs query → arbitrary-position lookup (retrieval, citation), pure SWA degrades. Mix in full-attention layers.
- **Window size is a real knob.** Too small: relevant context falls out of window. Too large: compute savings shrink. 1024–4096 are common defaults; production code models lean toward 4096.
- **KV-cache discard at the window boundary.** Once a token slides out of every layer's window, its KV entries can be dropped, but only after accounting for the longest layer-wise effective range. Naive eviction can break the receptive-field composition.
- **Attention sinks.** Some implementations pin the first few tokens (BOS, system tokens) inside every window to preserve attention-sink behavior. Skipping this can collapse decoding quality on long contexts.
- **Training-vs-inference window mismatch.** Some recipes train at a smaller window and extend at inference; the model can lose calibration. Train at the production window when possible.

---

## Sources

- Paper: *Longformer: The Long-Document Transformer* — Beltagy, Peters, Cohan, 2020 — early popularization of sliding-window attention in transformer LMs.
- Paper: *Mistral 7B* — Jiang et al., 2023 — production-scale validation of pure-SWA at window 4096.
- Paper: *Mellum 2* — JetBrains, 2026 — 3-of-4-layer mixed-SWA pattern with full-attention "express links" and layer-selective YaRN.
