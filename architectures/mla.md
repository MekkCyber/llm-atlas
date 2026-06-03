# Multi-head Latent Attention (MLA)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** An attention variant that **compresses the KV cache** by storing a single low-rank *latent* per token instead of per-head K and V. Heads reconstruct their keys and values on the fly via learned up-projections. A small **decoupled RoPE** slice handles positional encoding separately. Result: 5–10× smaller KV cache than standard multi-head attention at comparable quality, introduced in DeepSeek-V2 and used unchanged in DeepSeek-V3.

**Prereqs:** [attention](../fundamentals/attention.md), [multi-head-attention](multi-head-attention.md), [positional-encoding](../fundamentals/_positional-encoding.md)
**Related:** [transformer-block](transformer-block.md)

---

## What it is

Standard multi-head attention caches, for every token and every layer, one K and one V vector **per head**. For an LLM with $L$ layers, $H$ heads, and head dim $d_h$, the KV cache grows as $2 \cdot L \cdot H \cdot d_h \cdot \mathrm{seqlen}$ scalars per sequence — the dominant memory cost at long context.

MLA replaces this with: cache one **shared low-rank latent** per token plus a small **rotary slice**. At attention time, each head reconstructs its K and V from the latent via a learned up-projection. The reconstruction happens inside the attention kernel so decoder steps never materialize the full per-head KV.

For DeepSeek-V3, the technical report fixes **residual / embedding width** $d = 7168$, **head count** $H = 128$, and **per-head attention dim** $d_h = 128$. Unlike the usual vanilla-MHA convention $d_{\text{model}} = H \cdot d_h$, here $H \cdot d_h = 16384 \ne d$: MLA builds keys, values, and queries in head space of total width $H \cdot d_h$, then the attention block projects back to $d$ (same as in the paper's notation $h_t \in \mathbb{R}^d$). The per-token KV cache shrinks from $2 \cdot H \cdot d_h = 32768$ scalars (hypothetical full per-head K/V materialization at this $d_h$) to $d_c + d_h^R = 512 + 64 = 576$ scalars — a ~57× reduction per layer.

---

## How it works

### The two down-projections

At each layer, for input hidden state $h_t$:

$$
c_t^{KV} = W^{DKV} \cdot h_t \quad \text{shape } [d_c],\; d_c = 512 \text{ (KV latent)}
$$

$$
c_t^{Q} = W^{DQ} \cdot h_t \quad \text{shape } [d_c'],\; d_c' = 1536 \text{ (Q latent)}
$$

Here the **D** in $W^{DKV}$ and $W^{DQ}$ means **down-projection**: maps from full-width $h_t$ into the smaller latent dimensions $d_c$ / $d_c'$.

The KV latent $c_t^{KV}$ is what gets **cached**. The Q latent is recomputed each step; it only exists to save parameters in the query up-projection.

### The per-head up-projections

Each head reconstructs its "content" keys and values from the shared latent:

$$
K_t^{C,(h)} = W^{UK,(h)} \cdot c_t^{KV} \quad \text{shape } [d_h - d_h^R]
$$

$$
V_t^{(h)} = W^{UV,(h)} \cdot c_t^{KV} \quad \text{shape } [d_h]
$$

$$
Q_t^{C,(h)} = W^{UQ,(h)} \cdot c_t^{Q} \quad \text{shape } [d_h - d_h^R]
$$

Different heads have different up-projections, but they all read from the same $c_t^{KV}$. That's what makes the cache small.

### Decoupled RoPE

RoPE rotates Q and K by position-dependent angles. But RoPE and a linear up-projection $W^{UK}$ don't commute — if you rotate the latent $c_t^{KV}$ and then up-project, the effective rotation is head-dependent, which breaks RoPE's properties. If you rotate after the up-projection, you have to do the up-projection at every attention step (no cache).

MLA's fix: carry a **separate rotary slice** that is RoPE-rotated but not latent-compressed.

$$
K_t^{R} = \mathrm{RoPE}(W^{KR} \cdot h_t) \quad \text{shape } [d_h^R],\; d_h^R = 64 \text{ (shared across heads)}
$$

$$
Q_t^{R,(h)} = \mathrm{RoPE}(W^{QR,(h)} \cdot c_t^{Q}) \quad \text{shape } [d_h^R] \text{ (per-head)}
$$

$K_t^{R}$ is computed once per token and **shared across all heads** (MQA-style for the rotary part). It's stored in the cache alongside $c_t^{KV}$.

### Assembling the attention inputs

Each head's effective key and query are concatenations of content and rotary parts:

$$
K_t^{(h)} = [\, K_t^{C,(h)} ;\, K_t^{R} \,] \quad \text{shape } [d_h]
$$

$$
Q_t^{(h)} = [\, Q_t^{C,(h)} ;\, Q_t^{R,(h)} \,] \quad \text{shape } [d_h]
$$

$$
V_t^{(h)} = V_t^{(h)} \quad \text{shape } [d_h] \text{ (no RoPE on V)}
$$

So the concat is $(d_h - d_h^R) + d_h^R = d_h$, not $d_h + d_h^R$: the content vectors are *shortened* to leave room for the rotary slice within the same per-head budget.

Standard scaled dot-product attention then runs head-by-head:

$$
\mathrm{attn}_t^{(h)} = \mathrm{softmax}\!\left( \frac{Q_t^{(h)} \cdot {K^{(h)}}^\top}{\sqrt{d_h}} \right) \cdot V^{(h)}
$$

### What's actually cached

For sequence length $S$, per layer:

$$
\text{KV cache per token} = d_c + d_h^R = 512 + 64 = 576 \text{ scalars}
$$

$$
\text{vs standard MHA} = 2 \cdot H \cdot d_h = 2 \cdot 128 \cdot 128 = 32768
$$

$W^{UK}$ and $W^{UV}$ are *weights*, not cache — computed once, applied per attention step. $K_t^{R}$ is small enough that caching it alongside the latent is cheap.

### Fused kernel trick

Because $W^{UK}$ is applied every step, you can algebraically absorb it into the query projection:

$$
Q^{(h)} \cdot (K_t^{C,(h)})^\top = Q^{(h)} \cdot (W^{UK,(h)} \cdot c_t^{KV})^\top = (Q^{(h)} \cdot W^{UK,(h)}) \cdot (c_t^{KV})^\top = \tilde{Q}^{(h)} \cdot (c_t^{KV})^\top
$$

At decode time, you compute $\tilde{Q}^{(h)} = (W^{UK,(h)})^\top \cdot Q^{(h)}$ once and then do the dot product against $c_t^{KV}$ directly. Same result, but the $W^{UK}$ cost is paid per query, not per token in the KV cache. Same trick for V: absorb $W^{UV}$ into the output projection. Effective attention runs in the latent dimension, which is cheaper than head dimension for long contexts.

---

## Why it matters

- **KV cache is the serving bottleneck.** For long context (32K – 128K) at serving time, KV cache memory dominates. MLA's ~50× reduction per layer directly translates to 50× more concurrent sequences, or 50× longer context per sequence, at the same GPU memory.
- **Quality parity with full MHA.** DeepSeek-V2 ablations show MLA matches (and slightly beats) standard MHA at equal parameter count. The compression isn't a quality hit — it's structural redundancy removal.
- **Better than GQA / MQA at long context.** [GQA](multi-head-attention.md) groups heads to share K/V (e.g. 8 groups of 16 heads). MLA goes further — all heads share the same low-rank latent. Same category of win, larger magnitude.
- **Absorbed-projection trick keeps decode fast.** Done naïvely, reconstructing K and V on every step is expensive. The absorption trick makes effective decode happen in $d_c$ dim rather than $H \cdot d_h$ — usually a speedup.

---

## Gotchas & tricks

- **The decoupled RoPE isn't optional.** A first attempt that just puts RoPE on the up-projected K breaks the latent invariance — you'd have to materialize per-head K to rotate it, losing the cache win. The split into content + rotary is the whole trick.
- **Rotary slice is shared across heads.** All heads attend to the same $K_t^{R}$. This is MQA-like for the rotary half and head-specific for the content half. Don't try to give each head its own rotary slice — it doesn't help and breaks the cache math.
- **$d_c$ and $d_c'$ are different.** In DeepSeek-V3 the KV latent is 512 but the Q latent is 1536. The Q latent is compressed only to save parameters, not to reduce cache (Q is recomputed each step, not cached). Smaller Q latent hurts quality; 1536 is the point where they stopped.
- **Attention kernel must know about the latent.** You can't just swap MLA into a standard FlashAttention kernel. You need a kernel that does $Q \cdot (W^{UK} \cdot c^{KV})$ efficiently, or (better) one that applies the absorbed projections. DeepSeek open-sourced kernels; FlashInfer has MLA support.
- **RoPE dim $d_h^R$ choice.** 64 works; much smaller (16) visibly hurts long-context quality; larger (128) is fine but wastes cache. 64 seems to be the settled value.
- **Not just for inference.** MLA is used at training too — the cache savings don't matter during training (full sequence is materialized), but the parameter savings and the slight quality bump do.
- **Memory for $W^{UK}, W^{UV}$ grows with $H$.** One up-projection matrix per head. At $H=128$ this is a lot of parameters — but they're cheap compared to the FFN and don't dominate the model size.
- **Ports to video diffusion despite the spectral assumption failing.** *VideoMLA* (Yesiltepe et al., 2026) replaces per-head K/V in long-rollout video DiTs (Wan, HunyuanVideo) with a shared low-rank latent + decoupled 3D-RoPE positional key, cutting per-token KV memory by 92.7% per layer. Notably, pretrained *video* attention is **not** low-rank (99%-energy effective rank far above any practical latent dim), refuting the usual "MLA works because attention is naturally low-rank" framing. Instead, the MLA bottleneck *creates* a usable subspace through training — both spectral and random initializations occupy nearly the full rank budget from start and training preserves it. Reframes how MLA-style designs should be motivated and tuned in non-language modalities.

---

## Sources

- Paper: *DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model* — DeepSeek, 2024 — introduces MLA.
- Paper: *DeepSeek-V3 Technical Report* — DeepSeek, 2024 — reuses MLA with the same hyperparameters at larger scale.
- Implementation: DeepSeek's `DeepSeek-V3` repo and the FlashInfer MLA kernel.
- Background: *GQA: Training Generalized Multi-Query Transformer Models* — Ainslie et al., 2023 — the closest predecessor in the KV-reduction family.
- Cross-modal port: *VideoMLA: Low-Rank Latent KV Cache for Minute-Scale Autoregressive Video Diffusion* — Yesiltepe et al., 2026 — [arXiv:2605.30351](https://arxiv.org/abs/2605.30351). Surfaces the bottleneck-creates-rank mechanism noted in the gotcha above.
