# OPTIC — cross-layer latent VQ for per-layer embeddings
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Per-Layer Embeddings (PLE) in Gemma 3/4 E-series models are a growing fraction of parameters — **35.2% in Gemma 4 E4B**. OPTIC compresses PLE to ~**100×** by storing *one quantized code per token* and decoding to each layer's embedding through a small layer-specific affine decoder. At 100× compression it retains **97.8%** of the uncompressed model's performance across seven benchmarks and sits well above conventional per-layer quantizers, with substantial GPU-memory savings at near-equivalent latency.

**Prereqs:** [_number-formats.md](./_number-formats.md), [../fundamentals/README.md](../fundamentals/README.md)
**Related:** [../quantization/README.md](../quantization/README.md), [../quantization/fp8.md](../quantization/fp8.md), [../inference/README.md](../inference/README.md)

---

## What it is

PLE is Gemma's choice to give each layer its *own* input embedding table rather than share a single tied embedding across layers. It is parameter-expensive at scale: tens of millions of tokens × hidden dim × number of layers. Standard quantization (per-element FP4/INT4) touches each layer independently and leaves most of the PLE footprint on-disk.

OPTIC reframes PLE as **one shared VQ codebook across layers plus a layer-specific affine decoder**. Each token gets one codebook index, independent of layer. For each layer, a tiny affine map decodes the shared code to that layer's embedding.

## How it works

1. **Collect PLE outputs.** For each token, record its embedding at every layer.
2. **Fit a shared VQ codebook.** Cluster tokens by their *joint* multi-layer embedding profile; each code represents a bundle of per-layer embeddings.
3. **Fit per-layer affine decoders.** For each layer $\ell$, learn a small $A_\ell, b_\ell$ such that $A_\ell \cdot e_c + b_\ell$ reconstructs the layer-$\ell$ embedding for code $c$. Decoders are tiny relative to the raw table.
4. **Serve.** At inference, map a token to its code once; each layer applies its affine decoder on demand (fused with the subsequent op for low latency).

Compression ratio = ratio of raw PLE bytes to (code bytes per token + layer decoder sizes).

## Why it matters

- **PLE is a growing share of parameters** on modern on-device models (Gemma 3/4 E-series). Compressing it is a direct memory lever.
- **Beats conventional quantizers at extreme ratios.** 100× is well past where per-element quantization collapses.
- **Fits serving constraints.** Low per-layer latency overhead because the decoders are tiny; GPU-memory savings compound with other quantization choices.
- **Opens a new axis.** Rather than quantizing finer, OPTIC shows that reorganizing the storage layout (one code per token, decoded per layer) is a lossless-looking win.

## Gotchas & tricks

- **Specific to PLE-style architectures.** On models with tied cross-layer embeddings, the method's redundancy argument doesn't apply.
- **Codebook training needs calibration data.** A diverse token distribution is important — biased codebooks collapse rare tokens.
- **97.8% is averaged across seven benchmarks.** The paper reports per-task numbers; some tasks (math, hard retrieval) lose more at 100× than others.
- **Not a replacement for weight quantization.** OPTIC is for the embedding table; the rest of the model still needs its own quantization strategy.

## Sources

- Paper: *Extreme Compression of Per-Layer Embeddings via Cross-Layer Latent Vector Quantization* — Luo, Zhao, Sun, Shang, Cui, Eldar, Shen, MBZUAI (lead) / Westlake / Weizmann / UCL, 2026 — [arXiv:2610.31309](https://arxiv.org/abs/2610.31309).
