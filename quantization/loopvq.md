# LoopVQ — Sub-1-bit KV Cache Vector Quantization for Looped Transformers
*Depth — joint cross-step vector quantization of the KV cache for depth-shared / looped models.*

**TL;DR:** Looped transformers (Ouro, Huginn, Universal Transformer) run the same block $L$ times, multiplying KV-cache cost by loop depth. LoopVQ **vector-quantizes a channel's KV values jointly across loop steps** — the correlation between a channel's values across iterations is exactly what VQ exploits — and fine-tunes codebooks to reproduce full-precision outputs. Pushes KV cache to **sub-1-bit / element**: at **0.5 bits/element** beats prior KV VQ methods on Ouro and Huginn; combined with TaSQ, retains most of BF16 accuracy at **0.25 bits/element**. In SGLang: ~50% decode-latency drop at 32K context; **4.3× / 6.1× peak throughput** over BF16 on 1.4B / 2.6B Ouro.

**Prereqs:** [_number-formats.md](./_number-formats.md), [../inference/README.md](../inference/README.md)
**Related:** [fp8.md](./fp8.md) · [../architectures/multi-head-attention.md](../architectures/multi-head-attention.md)

---

## What it is

In a **looped transformer**, the same stack of layers is applied $L$ times to the same tokens — depth is "virtualized" by iteration. Each iteration produces its own K and V. The total KV cache is

$$
\text{size} \;=\; L \,\cdot\, 2 \,\cdot\, H \,\cdot\, d_h \,\cdot\, T
$$

($H$ heads, $d_h$ head dim, $T$ tokens). Looping gives you more effective depth at a parameter-count discount — but you still pay the KV cost of a model that deep.

Standard KV quantization (per-token scalar quantization, GPTQ-style, vector quantization per iteration) treats each iteration's K/V independently. But **the same channel's values across loop iterations are highly correlated** — the token hasn't moved, and the block is the same. LoopVQ codes those values *jointly*.

## How it works

**Joint cross-step codebook.** For a given channel $c$, head $h$, and token $t$, collect the $L$ values across loop iterations:

$$
\mathbf{v}_{t,h,c} \;=\; \bigl(v^{(1)}_{t,h,c}, v^{(2)}_{t,h,c}, \ldots, v^{(L)}_{t,h,c}\bigr) \in \mathbb{R}^L
$$

Quantize this $L$-vector against a per-channel codebook of $2^b$ centroids — one code per channel per token, replacing $L$ separate quantizations. The effective bits/element is $b / L$.

**Codebook fine-tuning.** Codebooks are initialized by k-means on a calibration set and then fine-tuned end-to-end against a reconstruction loss on the model's full-precision outputs — small, fast, no retraining of the base weights.

**Composes with TaSQ.** Task-specific residual quantization applied on top captures the quantization error and pushes the budget to 0.25 bits/element.

## Why it matters

KV cache is already the dominant memory cost at long context on every modern deployment. Looped models **amplify** that by $L$. LoopVQ shows the amplification **is the compression bonus** — the shared structure across iterations is exactly what a vector quantizer wants. The same trick applies to any **weight-sharing** regime that produces highly correlated per-step activations (e.g. parameter-shared attention variants).

## Gotchas & tricks

- Requires looped / depth-shared architectures to be interesting. On a plain stack, there's no cross-step correlation to exploit.
- Codebook size × channel count can grow if you per-channel it too aggressively; share codebooks across similar channels or use product quantization.
- Fine-tuning the codebooks is sensitive to the calibration distribution — long-context calibration is a must if the model will serve long contexts.
- Combined with KV paged attention, the per-page codebook indices become the new "KV page" — compatible with PagedAttention schedulers but changes the per-page metadata layout.

## Sources

- Paper: *LoopVQ: Sub 1-bit KV Cache Quantization for Looped Transformers* — Cheong, Yoo, 2026 — https://arxiv.org/abs/2611.05304
- Related: *Ouro* (depth-shared Transformer), *Huginn* (looped reasoning model), *TaSQ* (task-specific residual quantization).
