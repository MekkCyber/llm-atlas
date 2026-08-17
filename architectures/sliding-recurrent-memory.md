# Sliding recurrent memory (Maglev)

*Depth — a two-part transformer (full-attention prefiller + sliding-window decoder with recurrent K/V injection) that keeps O(1) decoding memory while training in parallel.*

**TL;DR:** Maglev splits the model into a **prefiller** with full attention that produces per-position memory K/V *targets*, and a **decoder** with sliding-window attention that predicts tokens using its short window *plus injected recurrent K/V*. A **memory-consistency loss** aligns the decoder's internal memory with the prefiller's target so training stays parallelizable. Result: fixed-memory decoding (like an RNN) with transformer-scale training throughput and parameter sharing that further compresses footprint.

**Prereqs:** [transformer-block](transformer-block.md), [multi-head-attention](multi-head-attention.md)
**Related:** [full-bandwidth-transformer](full-bandwidth-transformer.md), [mla](mla.md), [hybrid-linear-attention](hybrid-linear-attention.md)

---

## What it is

Classical sliding-window attention gives O(1) decoding memory but loses global context outside the window. Latent recurrent transformers restore some of that context via a compressed state but pay a training-parallelism tax. Maglev's insight: **train two networks jointly** — one that has global context and knows what the recurrent state *should* be, and one that has only the window and learns to *approximate* that state.

- **Prefiller** — full-attention model, sees the whole context, computes reference K/V per position.
- **Decoder** — sliding-window attention, additionally reads a fixed-size recurrent K/V that is *injected* from its previous step. Trained to match the prefiller's reference K/V at each position.
- **Consistency loss** — an auxiliary loss $\|K_{\text{dec}} - K_{\text{pref}}\|^2 + \|V_{\text{dec}} - V_{\text{pref}}\|^2$ (or a distributional variant) that ties the two.

At inference the prefiller is used once for prefill; the decoder handles autoregressive generation using only its window + recurrent state.

## How it works

Per training step:

1. Sample a batch of sequences.
2. Run the **prefiller** forward with full attention to produce reference K/V at every position.
3. Run the **decoder** forward with sliding-window attention + recurrent K/V injection (predicted from its own previous positions).
4. Standard LM loss on the decoder's logits.
5. Plus consistency loss aligning decoder K/V with prefiller K/V.

Parameter sharing between prefiller and decoder is optional and reduces memory footprint; the paper reports gains from sharing the FFN blocks while keeping attention distinct.

At inference:

- **Prefill:** run the prefiller over the prompt. Store the last recurrent K/V.
- **Decode:** advance the decoder step-by-step with sliding-window attention over the last $W$ tokens + injected recurrent K/V from the previous step. Memory footprint = $W \times d$ (window) + O(1) (recurrent state).

## Why it matters

- **Fixed-size decoding memory** without giving up parallel training — the two properties that were previously exclusive in this design space.
- **Improved downstream benchmarks over both plain sliding-window and latent recurrent transformers** at matched parameter budgets.
- **Composable with quantization and MLA** — the recurrent K/V injection is a small compressed tensor; the same tricks that quantize KV caches work here.

## Gotchas & tricks

- **Consistency loss weight is delicate.** Too low → decoder ignores the recurrent state; too high → decoder becomes a lossy copy of the prefiller and loses expressive capacity.
- **Prefiller/decoder capacity ratio.** Under-parameterizing either destabilizes training; the paper reports parameter sharing as the practical resolution.
- **Window choice matters.** Too-small $W$ makes the recurrent state overworked; too-large loses the O(1) advantage.
- **Not a long-context recall panacea.** The recurrent state is a lossy compression of the past; needle-in-a-haystack tasks that require exact recall of arbitrary past tokens still favor a full KV cache or hybrid.

## Sources

- Paper: *Maglev: Sliding Recurrent Memory* — Bo Liu, Qiang Liu (UT Austin), 2026, [arXiv:2608.02870](https://arxiv.org/abs/2608.02870)
