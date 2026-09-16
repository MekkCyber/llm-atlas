# Grouped Value Attention (GVA)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A KV-cache shrinking attention variant that **caches only values plus a small positional key** and reconstructs content keys on demand from cached values via a learned linear map. At decode the reconstruction map is absorbed into the query so content keys never materialize on the hot path. In the 350M-parameter reference configuration this cuts persistent cache scalars by ~45–47% vs matched GQA at essentially the same downstream accuracy.

**Prereqs:** [multi-head-attention](multi-head-attention.md), [rope](../fundamentals/rope.md)
**Related:** [mla](mla.md), [_attention-variants](_attention-variants.md)

---

## What it is

GQA and MQA reduce the KV cache by **sharing K and V heads across groups of queries**. MLA reduces it by **caching a shared latent** and reconstructing per-head K and V. GVA takes a third route: **cache values only**, and derive content keys from the same value cache at attention time. A small **decoupled RoPE key channel** carries positional information separately, mirroring the split MLA uses.

## How it works

Per token, per layer, three things are computed:

- A **grouped value** vector $V_t \in \mathbb{R}^{d_v}$ — this is what gets stored persistently.
- A small **positional key** $K_t^R = \mathrm{RoPE}(W^{KR} h_t) \in \mathbb{R}^{d_h^R}$ shared across heads — also stored, but cheap.
- **No content key is stored.** Instead, at attention time each head reconstructs its content key via a learned linear map from the cached value: $K_t^{C,(h)} = W^{VK,(h)} V_t$.

The attention dot product at decode is $Q^{(h)} \cdot [K^{C,(h)} ; K^R]^\top$. The $W^{VK}$ multiplication can be algebraically absorbed into the query so decoding computes $(Q^{(h)} W^{VK,(h)}) \cdot V_t^\top$ against the value cache directly, and the RoPE contribution runs against $K^R$. That absorption is the same trick that keeps MLA decode fast.

## Why it matters

- **~45–47% smaller persistent KV cache** than matched GQA in the reference 350M configuration, with average accuracy on 5 tasks essentially at GQA parity (44.18 vs 44.36, MLA at 43.88 in the same setup).
- Adds a third orientation to the KV-compression family: GQA/MQA share heads, MLA caches a latent, **GVA drops the content-key store entirely**. The wins are not obviously stackable but they are structurally different, so future work can look for combinations.
- The absorption trick means the memory win does not cost a decode-speed penalty in principle; the paper announces custom kernels and an open release rather than shipping numbers on end-to-end latency yet.

## Gotchas & tricks

- **Decoupled-RoPE key is not optional.** RoPE and the reconstruction map $W^{VK}$ do not commute, so rotating the reconstructed content key breaks the absorption. The rotary information has to ride in a separately-cached small channel — same failure mode MLA solves the same way.
- **Reconstructing keys from values is a strong prior.** It only works if K and V of a given token are approximately linearly related in the trained model; the paper's ablations focus on 350M-scale results and this remains to be tested at frontier scale.
- **"Persistent scalars" is what shrinks.** Peak activation memory during attention (which materializes reconstructed content keys momentarily per step) is not what changes — the paged/prefill numbers, and the decode-time in-flight tensors, need to be reported separately.
- **Kernels are not upstreamed at time of writing.** Adoption depends on the announced release; without custom kernels, naive PyTorch attention will re-project the content keys per step and lose the compute win.

## Sources

- Paper: *Grouped Value Attention: Efficient KV Caching via On-Demand Key Reconstruction* — Tripathi, Kumar, Khan — 2026 — [arXiv:2609.13285](https://arxiv.org/abs/2609.13285).
- Background: *DeepSeek-V2 / V3* — DeepSeek — 2024–2025 — introduces MLA, the closest structural relative.
