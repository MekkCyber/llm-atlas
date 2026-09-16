# Sliding-Window Attention
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Restrict attention so each token only attends to the previous $W$ tokens (typically $W \in \{256, 4096\}$). The KV cache still stores every token, but effective attention runs over a bounded window per layer, so both compute and cache-read traffic per decode step become $O(W)$ instead of $O(\mathrm{seqlen})$. Long-range information passes through the stack layer by layer, extending an effective receptive field of roughly $L \cdot W$ across $L$ sliding-window layers. Modern bases interleave sliding-window layers with a few **full-attention** layers to keep long-range fusion sharp.

**Prereqs:** [attention](../fundamentals/attention.md), [multi-head-attention](multi-head-attention.md)
**Related:** [_attention-variants](_attention-variants.md), [mla](mla.md)

---

## What it is

Standard causal attention lets each token attend to every earlier token. That is $O(n^2)$ FLOPs and, more critically for decoding, $O(n)$ cache reads per step. Sliding-window attention caps each token's attention to a fixed window $W$ of most-recent tokens; older keys remain in the cache but are masked out.

Historically deployed in Longformer / Mistral / early Llama-3 8B variants; recent small-model tech reports revive it as a way to make **compact bases** viable at long context when paired with tool use or retrieval.

## How it works

The change is a mask. For token at position $t$ attending to positions $i \le t$:

$$
\mathrm{mask}(t, i) = \begin{cases} 0 & t - W < i \le t \\ -\infty & \text{otherwise} \end{cases}
$$

At training time the mask makes softmax ignore anything older than $W$ tokens; at inference the same mask means only the last $W$ keys need to be read. Attention kernels (FlashAttention 2/3, xFormers block-sparse) support the window mask natively.

**Interleaving with full attention.** A common pattern is $N_\text{sw}$ sliding-window layers followed by one full-attention layer, repeated. Each full-attention layer re-fuses information across the whole context, letting long-range signal escape the window. Choice of ratio (e.g. 3:1, 5:1) trades throughput for long-range quality.

**RoPE inside the window.** Positional information still comes from RoPE; sliding-window does not change that. The window just decides which key-position pairs are visible.

## Why it matters

- **Cache-read cost caps out.** Per decode step, the attention read is $O(W)$ regardless of sequence length. At $W = 4096$ this is what makes 256K-context serving on 7B-class models tractable.
- **Compact bases become long-context.** For small models the parametric cost of true global attention over 256K+ tokens is prohibitive; sliding-window keeps most layers local, and the few full-attention layers do the long-range integration.
- **Composes with cache-shrinking variants.** Sliding-window sits on the *reach* axis and cache-shrinking variants (GQA, MLA, GVA) sit on the *per-token size* axis, so they compose. The 2026 ZGCM-1 7B/256K-context base uses this pattern.

## Gotchas & tricks

- **Long-range information vanishes without full-attention layers.** Pure sliding-window networks cannot propagate information further than $L \cdot W$ tokens in the best case, and in practice much less because signal is diluted through nonlinearities. Interleaved full-attention layers or state-space layers are almost always needed.
- **Window boundary drift at training/serving mismatch.** If training uses a shorter window than serving (e.g. 2048 vs 4096) the network's implicit prior about window size can bite. Match them, or fine-tune to the serving window.
- **Chunked prefill needs a window-aware kernel.** Naive chunked prefill leaves the mask stale at chunk boundaries. FlashAttention chunked-prefill kernels handle this if the window is a config parameter.
- **Retrieval / tool use is the intended complement.** Sliding-window bases lean on external retrieval for content outside the window; using them without that pattern usually leaves capability on the table.

## Sources

- Paper: *Longformer: The Long-Document Transformer* — Beltagy, Peters, Cohan — 2020 — introduces sliding-window attention with a global-token variant.
- Model: *Mistral 7B* — Jiang et al. — 2023 — sliding-window attention in a modern LLM base.
- Recent: *ZGCM-1* — 2026 — [arXiv:2609.13356](https://arxiv.org/abs/2609.13356) — mixes sliding-window with full-attention layers in a 7B compact-reasoner base at 256K context.
