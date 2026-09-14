# Hybrid-Precision Attention Quantization
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Quantize most attention states to low-bit but keep a small set of **vertical-line tokens (attention sinks)** and **local-window** states in high precision. Applies at both prefill and decode. Achieves near-lossless accuracy at low bit-widths where uniform quantization collapses. HyQuant is the concrete instantiation.

**Prereqs:** [_number-formats](_number-formats.md), [../fundamentals/attention](../fundamentals/attention.md)
**Related:** [fp8](fp8.md), [../inference/README](../inference/README.md)

---

## What it is

Uniform low-bit quantization of attention (Q, K, V or the KV cache) is fragile: a handful of tokens carry disproportionate attention mass — the well-known *vertical-line tokens*, also called attention sinks or heavy hitters — and the sliding local window carries most fine-grained signal. Quantizing these two hot sets the same way as the rest destroys accuracy.

Hybrid-precision attention quantization splits the attention state:
- **Hot set (high precision):** vertical-line tokens (few, persistent, sequence-wide sinks) + a local window around the current position.
- **Cold set (low precision):** everything else — the long tail of the KV cache and the non-local attention states.

Both sets participate in the same attention math; the only change is the precision of the operands.

## How it works

- **Sink identification.** At prefill, detect the vertical-line tokens by their attention-mass profile (they accumulate mass from all query positions). Mark them for the hot set. This is cheap — a running-max over per-column attention weights.
- **Local-window hot set.** Keep the last $W$ tokens of the KV cache in high precision. $W$ is the sliding-window size (typical 256–1024).
- **Cold set low-bit storage.** All other KV entries are stored in low-bit format (INT4/FP4/MXFP4-class), dequantized on the fly at attention time.
- **Same math, mixed operands.** The attention softmax and value-mix are performed in the higher precision; low-bit values are upcast per operation. No custom softmax kernel needed.
- **Prefill and decode symmetry.** The scheme is applied at both stages so latency and memory wins compound: prefill benefits from smaller K/V footprints; decode benefits from fewer bytes moved through HBM per new token.

## Why it matters

Attention has been the sticking point for aggressive KV-cache quantization — every prior attempt at INT4 KV loses noticeable accuracy on long contexts. Recognizing that attention sinks + local windows are the real bottleneck and carving them out of the low-bit path is the natural unification of two known observations (sink phenomena + local-window bias). This unblocks cheap long-context inference without retraining.

## Gotchas & tricks

- **Sink detection must be online.** Which tokens become sinks depends on the prompt; a static sink set (e.g. always the first token) misses most sinks in real workloads.
- **Local-window size is a knob.** Too small (< 128) and quality drops on local grammar / structure; too large (> 2048) and the hot-set memory bloats.
- **Not the same as StreamingLLM.** StreamingLLM *drops* the middle of the KV cache; hybrid-precision quantization *keeps* it, only at lower precision.
- **Kernel cost.** The mixed-precision attention kernel needs to interleave dequant with dot products; naive implementations lose the memory-bandwidth win to launch overhead.
- **Compatible with weight quantization.** Orthogonal to weight-only INT4 (GPTQ/AWQ) — you can stack both.

## Sources

- Paper: *HyQuant: Hybrid-Precision Quantization for LLM Attention* — Ding, Xing, Zhang, et al. (Shanghai Jiao Tong University), 2026 — https://arxiv.org/abs/2608.27875
- Related: attention-sink and vertical-line-token literature (Xiao et al., 2023; Liu et al., 2024).
