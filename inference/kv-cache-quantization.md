# KV Cache Quantization
*Depth — compressing the attention key/value cache to reduce the dominant memory cost of long-context serving.*

**TL;DR:** The KV cache is the per-token snapshot of attention keys and values that lets subsequent decode steps avoid recomputing attention over the prompt. At long contexts it dominates GPU memory capacity *and* bandwidth, so serving throughput is roughly inversely proportional to its byte size. Quantizing K and V to FP8, FP4, or INT8 — with a scale granularity chosen to protect attention probabilities — gives 2–4× memory wins at a small accuracy cost. The modern recipe is per-head, per-block scales; the frontier is **native FP4 KV** on Blackwell (BitDecoding-FP4, 2026).

**Prereqs:** [README.md](README.md), [../quantization/_number-formats.md](../quantization/_number-formats.md)
**Related:** [../quantization/fp4.md](../quantization/fp4.md), [../quantization/fp8.md](../quantization/fp8.md)

---

## What it is

Every transformer decode step reads the full history's K and V tensors to compute attention. Those tensors live in a per-request buffer (the KV cache) that grows linearly with context length and dwarfs the model weights at long contexts. Example: for a 70B model at 128K context, KV cache is often 2–4× the size of the weights, not the other way around.

KV cache quantization stores K and V in a narrower numeric type (FP8 / FP4 / INT8 / INT4 / NF4) with a reconstruction scale, trading a small accuracy hit for a proportional memory win. Unlike weight quantization (done once, offline), KV quantization has to run *online* per token as the context extends, which constrains what calibrations are feasible.

## How it works

Each new K and V vector produced during prefill or decode is quantized before going into the cache:

```
K_quant, s_K = quantize(K_fp, block_size)    # e.g. FP4 with 32-element blocks
V_quant, s_V = quantize(V_fp, block_size)
cache[t] = (K_quant, s_K, V_quant, s_V)
```

Attention at decode reads back cached K/V, dequantizes inline in the kernel, and computes the standard `softmax(QK^T/√d) · V`. Three design choices:

- **Block / group scale granularity.** Per-head per-block (e.g., 32 or 128) is the sweet spot. Per-tensor scales are too coarse (outliers in one head poison everything). Per-element is as big as not quantizing.
- **Protect the softmax.** Attention probabilities are highly sensitive to K-side error because small logit perturbations exponentiate. Many practical schemes leave queries in BF16 or FP8 even when K/V go to FP4.
- **Paged layout.** Quantization must be compatible with paged attention (vLLM-style): blocks are the same unit for quantization scales and for the page table.

Variants: INT8 (SmoothQuant-style), FP8 (vLLM / TensorRT-LLM production), KIVI (INT4 asymmetric), NF4 with QLoRA-style quantile levels, and most recently MXFP4 / NVFP4 with native hardware support.

## Why it matters

Long-context serving (agent sessions, document QA, code copilots with repo context) is essentially a KV-cache memory-bandwidth problem. Halving the KV cache nearly doubles serving throughput per GPU. BitDecoding-FP4 on NVIDIA B300 reports **2.4–3.0× higher peak serving throughput than BF16 KV** on Qwen3 at **≤0.5-point accuracy drop (NVFP4)** — the biggest single lever available right now on long-context inference.

## Gotchas & tricks

- Key quantization error hurts far more than value quantization error at the same bit width. Many schemes use asymmetric budgets (K at a higher precision than V).
- Prefix caching + KV quantization interact: reusable prefixes should be re-quantized in-place with consistent scales, or the cached scales become per-request memory overhead.
- Initial tokens (the first ~4–8 positions, "attention sinks") are disproportionately important; keep those in higher precision even if the rest is FP4.
- Measure both acceptance-style metrics (perplexity, MMLU) and *generation* metrics (long-context retrieval, agent-task completion). Cache quantization can preserve the former while meaningfully degrading the latter.
- The accuracy cliff for INT4 or FP4 appears earlier for long outputs than short ones — small per-step errors compound.

## Sources

- Paper: *BitDecoding-FP4: Native FP4 KV Cache Decoding via Algorithm–System Co-Design* — Du et al., 2026 — [arXiv:2611.09825](https://arxiv.org/abs/2611.09825).
- Paper: *KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache* — Liu et al., 2024 — [arXiv:2402.02750](https://arxiv.org/abs/2402.02750).
- Paper: *SmoothQuant* — Xiao et al., 2022 — INT8 activation/weight quantization patterns that transferred to KV.
- vLLM docs — FP8 KV cache in production serving.
