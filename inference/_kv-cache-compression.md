# KV Cache Compression
*Taxonomy — strategies for shrinking the per-token key/value memory footprint at inference time.*

**TL;DR:** Long contexts and long reasoning outputs explode the KV cache — for production reasoning models, the KV is often the dominant memory cost and the binding constraint on batch size. Compression splits into three orthogonal levers: **what each token stores** (smaller per-token KV — MLA, GQA, MQA), **which tokens stay in cache** (eviction — H2O, SnapKV, InfoKV), and **how cached tokens are encoded** (quantization — KV INT4/INT8). All three compose. The 2026 frontier on the eviction axis is **information-theoretic signals beyond attention** — the recipe InfoKV ships.

**Related taxonomies:** [_speculative-decoding](_speculative-decoding.md) · [../quantization/_number-formats](../quantization/_number-formats.md)
**Depth files covered here:** [infokv](infokv.md)

---

## The problem

KV cache size grows as `layers × heads × head_dim × 2 × sequence_length × batch`. For a 70B Llama-class model at 32k context and batch 16, the KV cache alone runs into hundreds of GB. Two production-relevant pain points:

- **Long-CoT reasoning outputs** push sequence_length into the tens of thousands and dominate end-to-end latency.
- **Agent traces** (multi-turn tool calls, accumulated tool outputs) extend prefill across the entire history.

If you can't shrink the KV per token or per cache, you can't grow the batch or the context. Compression is the only knob.

## The shared pattern

Every KV compression scheme answers the same question: *for each token in the cache, do we keep its full KV, a smaller version, or nothing*?

| Axis | Question | Where it acts |
| --- | --- | --- |
| **Per-token KV size** | How big is one token's stored KV? | Architecture (training time) |
| **Eviction** | Which tokens stay in cache? | Inference time, mid-decode |
| **Quantization** | What numeric format encodes the kept tokens? | Inference time, all positions |

The levers are independent and stackable: MLA + InfoKV + INT4 KV all at once is fine.

## Variants

| Technique | Axis | Mechanism | Tradeoff |
| --- | --- | --- | --- |
| MQA / GQA | Per-token size | Heads share K/V | Quality drop at extreme sharing |
| **[MLA](../architectures/mla.md)** | Per-token size | Low-rank KV projection | Requires training the projection |
| StreamingLLM | Eviction | Keep recent + attention-sink tokens | Loses non-recent informative content |
| H2O | Eviction | Heavy-Hitters: keep top-attention tokens | Local attention only — misses distant influence |
| SnapKV | Eviction | Query-aware: keep tokens the recent query attended to | Tied to query window; less effective on long-CoT |
| **[InfoKV](infokv.md)** | Eviction | Attention scores + token-level entropy (Forward Influence) | Needs the model's own probabilities |
| KV INT4 / INT8 | Quantization | Quantize stored KV tensors | Calibration noise; pairs with PagedAttention layouts |

## How to choose

- **Architectural lever first.** If you control the training stack, [MLA](../architectures/mla.md) (or GQA at minimum) is the cheapest 2-5× cache reduction available. Per-token size beats post-hoc eviction.
- **For long-CoT reasoning workloads**, attention-only eviction (H2O, SnapKV) underperforms — pick [InfoKV](infokv.md) or another method that incorporates non-local signals (predictive uncertainty, layer-wise evolution).
- **For multi-turn agent workloads**, query-aware eviction (SnapKV) is a strong default since most queries reference recent tool outputs.
- **Quantization** is the easiest production lever and composes with everything else; calibrate carefully on representative inputs.
- **Stack the levers** — MLA + InfoKV + INT4 KV all at once is standard for 2026 long-context deployments.

## Adjacent but distinct

- **PagedAttention / vLLM** — a *layout* optimization for the KV cache, not a compression technique. Required infrastructure for any cache-management scheme to work in production.
- **Prefill/decode disaggregation** — splits the workload across machines so prefill KV doesn't compete with decode KV. Orthogonal to compression.
- **[Speculative decoding](_speculative-decoding.md)** — reduces decode steps; KV compression reduces per-step cost. Compose for compounding speedups.

## Sources

- Paper: *Efficient Streaming Language Models with Attention Sinks (StreamingLLM)* — Xiao et al., 2023.
- Paper: *H2O: Heavy-Hitter Oracle for Efficient Generative Inference of LLMs* — Zhang et al., 2023.
- Paper: *SnapKV: LLM Knows What You Are Looking for Before Generation* — Li et al., 2024.
- Paper: *DeepSeek-V2 / V3 — Multi-Head Latent Attention* — DeepSeek, 2024–2025.
- Paper: *Information-Aware KV Cache Compression for Long Reasoning* — Xiao, Birch, Lin, 2026 — [arXiv:2606.26875](https://arxiv.org/abs/2606.26875).
