# KV Cache Quantization
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** The KV cache dominates long-context inference memory and bandwidth. Quantizing it to 2–4 bits (vs. fp16) multiplies the context window per unit memory and speeds up attention reads. The core trick is choosing a pre-quantization basis that minimizes reconstruction error on the two-sided matrix product that attention computes. WUSH-KV (Chen et al., IST Austria / ETH, 2026) builds a data-adaptive transform from second-order statistics of the attention product and shows near-optimality under clipped quantizers like QuEST-INT.

**Prereqs:** [_number-formats.md](./_number-formats.md), [../fundamentals/rope.md](../fundamentals/rope.md).
**Related:** [fp8.md](./fp8.md).

---

## What it is

Autoregressive attention reads

$$
y = \text{softmax}(Q K^\top) V
$$

For every new query $Q$, the model reads back the full $K$ and $V$ tensors it cached from previous tokens — size $\propto$ context length $\times$ heads $\times$ head-dim per layer, in fp16. For a 128K context, this is tens of GB per request. Quantizing $K$ and $V$ to 2–4 bits (while leaving the fp16 Q) is the main lever.

Error introduced by quantizing $K$ and $V$ appears in a specific place: the attention matrix product $Q K^\top$ and the weighted-sum $(\text{softmax} \cdot V)$. Minimizing *that* error, not per-tensor reconstruction error, is what distinguishes good KV quantization schemes.

---

## How it works

**Rotate-then-quantize.** Classical GPTQ-style trick: apply an orthogonal transform $R$ before quantizing so the resulting distribution is better matched to the quantizer grid. Common choices: Hadamard (data-free), learned rotations, or the WUSH transform (data-adaptive).

**WUSH transform.** Constructed from the **second-order statistics of both factors** in the matrix product being quantized. For attention, that means using calibration data to compute $\mathbb{E}[K K^\top]$ and $\mathbb{E}[V V^\top]$ (roughly), then picking a basis that balances error contributions from the two sides.

**Implementation details.**
- Separate transforms for $K$ and $V$ (different statistics).
- $V$ transform **folds into model weights**: $W_V' = R_V \cdot W_V$, so there's no inference-time overhead on the value side.
- $K$ transform **applies after RoPE**: RoPE operates on $K$ in its native basis; the transform follows.
- Pairs with a clipped quantizer like QuEST-INT (per-channel scales + outlier clipping) — the paper shows near-optimality under mild assumptions.

**Deployment.** WUSH-KV integrates into SGLang using OSCAR-style percentile-clipped affine quantization at the serving layer.

---

## Why it matters

- **2-bit KV at perplexity-equivalent quality.** At 2-bit, WUSH-KV matches or beats OSCAR across all evaluated models and tasks; near-optimal in a formal sense for QuEST-INT.
- **Serving economics.** Long-context inference is KV-cache-bound; 2-bit vs. fp16 is an 8× memory reduction and a corresponding bandwidth saving.
- **Clean integration with RoPE.** Earlier KV-quant schemes either disabled RoPE or quantized the un-rotated $K$ (which hurt). Applying the transform *after* RoPE avoids both pitfalls.
- **Theory meets practice.** The near-optimality proof is a sanity check that the transform is doing what rotation-based quantization has been trying to do heuristically.

---

## Gotchas & tricks

- **Calibration data matters.** Second-order statistics are estimated from a calibration set; sensitive to distribution shift (e.g. coding vs. chat). Calibrate on data representative of deployment.
- **RoPE ordering is load-bearing.** Applying the $K$ transform *before* RoPE scrambles the positional phase relationships; always apply after.
- **$V$ transform folds for free.** $K$ transform does not — it must run at inference time on every new key. Keep it a cheap matmul, not a full dense layer.
- **Per-head vs. per-layer transforms.** Finer granularity helps quality but costs memory for the transform matrices. Common choice: per-layer.
- **MoE caveat.** Each expert can have a different $K/V$ statistics distribution; a global transform underfits. Per-expert transforms are an open area.
- **Doesn't eliminate attention cost.** You still read 2-bit tensors — bandwidth is lower, but the softmax and matmul still run.

---

## Sources

- Paper: *WUSH-KV: KV Cache Quantization with Data-Adaptive Transforms* — Chen, Egiazarian, Kurtić, Hoefler, Alistarh, 2026. IST Austria / ETH Zurich. arXiv:2609.38121.
- Related: *OSCAR* — percentile-clipped affine KV quantization baseline WUSH-KV compares to.
- Related: *QuEST-INT* — clipped integer quantizer paired with WUSH transforms in the paper.
