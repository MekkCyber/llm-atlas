# FP8 Mixed-Precision Training
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** End-to-end pretraining where forward, backward, and most intermediate tensors live in **FP8** — specifically E4M3 with **fine-grained per-tile scaling** (1×128 for activations, 128×128 for weights), paired with a **higher-precision accumulation** trick that promotes partial sums to FP32 every 128 inner-dim elements. Certain components (embeddings, LM head, MoE gating, norms, attention) are kept in BF16/FP32. Master weights stay FP32, optimizer moments go to BF16. First validated at frontier scale by DeepSeek-V3 (671B total / 37B active, 14.8T tokens).

**Prereqs:** [fp8](../quantization/fp8.md), [_number-formats](../quantization/_number-formats.md), [_training-stability](_training-stability.md)
**Related:** [mid-training](mid-training.md)

---

## What it is

The dominant training dtype since 2020 has been BF16 for compute with FP32 master weights. FP8 on hardware (Hopper+) offers 2× throughput and 2× memory over BF16 — but the naive recipe NVIDIA published (per-tensor scaling, E4M3 forward / E5M2 backward, hybrid schedule) is fragile at LLM scale: activation outliers overflow single-scale tensors, gradients drift, and loss spikes mid-run.

DeepSeek-V3's FP8 framework is the first public recipe that trained a frontier-scale model end-to-end in FP8 without stability surgery. It sits on three changes: **fine-grained quantization tiles**, **increased-precision accumulation**, and **selective high-precision components**.

This file covers the training-specific side. The FP8 format itself (E4M3, E5M2, value formula, matmul semantics) is in [fp8](../quantization/fp8.md).

---

## How it works

### Which tensors are FP8 and which are not

**FP8 (E4M3):**
- Forward matmul inputs: activations and weights.
- Backward matmul inputs: upstream gradients and weights (for `∂L/∂x`), upstream gradients and activations (for `∂L/∂W`).
- Cached activations for backward — where safe.

**BF16 / FP32:**
- Embedding layer weights and LM head weights (tight numerics, small FLOP share, sensitive to rounding).
- RMSNorm / LayerNorm gains, and the normalization operations themselves.
- All attention operations — softmax, scaling, masking (highly sensitive to quantization error).
- MoE gating / router (small and high-variance — quantization hurts routing).
- Master weights (FP32).
- Gradients after accumulation, before optimizer step (BF16 or FP32).
- Optimizer moments `m`, `v` — BF16 in DeepSeek-V3 (vs FP32 standard), saves ~2× optimizer state memory.

Rule of thumb: FP8 for the big matmuls (attention's Q/K/V/out projections, FFN up/gate/down), BF16/FP32 for everything numerically delicate.

### Fine-grained quantization tiles

A single scale per tensor is the fragile part of the NVIDIA recipe. DeepSeek uses **per-tile** scales:

```
Activations (shape [B·T, d]):
    one scale per 1 × 128 tile   ← per (token, 128-channel group)
    scale storage: [B·T, d/128] in FP32

Weights (shape [d_out, d_in]):
    one scale per 128 × 128 block
    scale storage: [d_out/128, d_in/128] in FP32
```

The finer granularity means an outlier in one channel group only wrecks its own scale — all other tiles keep their dynamic range intact. E4M3 throughout becomes viable because each tile's local range is well-bounded.

**Scale computation (forward):**

```
for each tile T:
    s_T = max(|T|) / 448           ← 448 = max E4M3
    T_fp8 = round(T / s_T)         ← store in FP8
```

The scale stays in FP32 and is carried alongside the FP8 payload.

### E4M3 everywhere

Standard advice is E4M3 forward, E5M2 backward (gradients need wider range). DeepSeek uses **E4M3 on all tensors** including gradients. This works because fine-grained tile scaling gives each tile enough local dynamic range that E4M3's limited exponent (4 bits, bias 7) doesn't bottleneck — and E4M3's extra mantissa bit gives measurable accuracy win vs E5M2.

### Increased-precision accumulation

Tensor cores on Hopper accumulate matmul partial sums in their native precision — which, for FP8 inputs, is BF16-ish. Over a long inner dimension (a 7168-wide matmul accumulates 7168 FP8 products) this introduces significant rounding error that compounds across layers.

DeepSeek's fix:

```
for each 128-element chunk along the K (inner) dimension:
    partial = tensor_core_matmul(A_chunk, B_chunk)   ← accumulates inside tensor core
    accum_fp32 += partial                            ← CUDA core, FP32 add
```

Every `N = 128` inner-dim elements, the tensor core's partial accumulator is read out into **FP32 registers in CUDA cores**, added to a running FP32 total, and the tensor-core accumulator is re-zeroed.

The cost: the CUDA-core add is slower than the tensor-core MAC, but it's only done every 128 elements, so the marginal cost is a few percent of total matmul time. The gain: accumulation error is now bounded by 128-element rounding, not 7168-element rounding.

### Optimizer and memory

Master weights, weight gradients (after reduction), and weight decay operations are all in **FP32**. The AdamW first moment `m` and second moment `v` are stored in **BF16**. Storing `m, v` in BF16 instead of FP32 halves optimizer memory — for a 671B-parameter model, that's ~1.3 TB saved across the cluster.

No measurable quality impact from BF16 optimizer moments in DeepSeek-V3's ablations. The underlying reason: AdamW is already numerically robust (ratios of moments cancel many precision effects), and BF16's 8-bit exponent covers the dynamic range of `v`.

### What stays high-precision

Explicit list from the paper:

```
embedding layer          → BF16/FP32
output (LM) head         → BF16/FP32
MoE gating / router      → BF16/FP32
RMSNorm / LayerNorm      → FP32 (input cast to FP32, norm applied, cast back)
attention op (softmax,
 mask, scaling)          → BF16/FP32
master weights           → FP32
accumulated gradients    → FP32
optimizer moments m, v   → BF16
```

Roughly: everything that is either (a) a small FLOP share, or (b) numerically sensitive.

---

## Why it matters

- **Frontier-scale FP8 training is proven.** DeepSeek-V3 is the first openly-documented model at 671B total params trained end-to-end in FP8 without stability surgery.
- **~2× memory, ~2× throughput** over BF16 for the main matmuls, at comparable quality.
- **~50% training cost reduction** vs a hypothetical BF16 reimplementation of the same recipe. The 2.788M H800-hour total training cost for DeepSeek-V3 wouldn't be achievable in BF16 on the same hardware.
- **Sets the template** for frontier FP8 training: fine-grained tiles, E4M3 everywhere, accumulation promotion, selective high-precision components. Subsequent frontier-scale FP8 work largely follows this shape.

---

## Gotchas & tricks

- **Don't skip the accumulation promotion.** The biggest stability win comes from periodically flushing to FP32. If your kernel uses tensor-core-native accumulation for the full inner dim, you'll see slow loss drift that looks like a learning-rate problem but isn't.
- **MoE gating in FP8 is a footgun.** The router has tiny logits near each other; FP8 rounding collapses them into ties. Keep the router in BF16.
- **Attention in FP8 needs care.** The softmax operation is numerically brittle. DeepSeek-V3 keeps the Q·K^T, softmax, and ·V chain in BF16/FP32. FP8 *matmul* for Q/K/V and output projections is fine — it's the attention operation inner loop that's delicate.
- **Embedding quantization kills small models.** Small vocab + FP8 embeddings = a handful of common tokens dominating the scale. Keep embeddings in BF16.
- **Tile boundaries matter for loading.** 1×128 activation tiles assume tokens are the row axis and channels are the column axis. If your tensor layout is different, tile shape needs to rotate — getting this wrong means scales apply to the wrong groups and training silently collapses.
- **Scale storage is not free.** 128×128 block scales on a 7168×7168 weight matrix are 56×56 = 3136 FP32 scales (~12 KB) — negligible. 1×128 activation scales on a `[B·T=4M, d=7168]` activation tensor are 4M × 56 = ~224M FP32 scales — adds up to a real fraction of activation memory if you're not careful about caching.
- **FP8 gradient all-reduce.** Gradients can be all-reduced in BF16 to avoid re-quantizing to FP8 across the ring. Doing the all-reduce in FP8 loses precision rapidly.
- **Stability is verified at scale, not at small scale.** 100M-param FP8 runs are easy. 100B-param FP8 runs expose failure modes (activation outliers in certain FFN-down projections, attention-logit drift interacting with FP8 attention) that small-scale ablations don't. If you're building a new FP8 framework, validate on the biggest model you intend to use it for.
- **Hopper-specific.** The fine-grained-tile recipe is tuned for Hopper's tensor cores. Blackwell's native MXFP8 path (32-element block scale in E8M0) is a different code path — the high-level recipe translates but the quantization details don't.

---

## Sources

- Paper: *DeepSeek-V3 Technical Report* — DeepSeek, 2024 — the canonical modern FP8 training recipe (§3.3).
- Paper: *FP8 Formats for Deep Learning* — Micikevicius et al., 2022 — E4M3 / E5M2 spec (background).
- Paper: *Using FP8 for Deep Learning Training* — Sun et al., 2020 — early FP8 training exploration.
- NVIDIA Transformer Engine documentation — the stock Hopper FP8 recipe this improves on.
- OCP Microscaling Formats (MX) v1.0 — for the Blackwell-era MXFP8 alternative.
