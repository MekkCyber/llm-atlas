# Linear Attention
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** An attention family that replaces the softmax dot-product with a **kernel-linearizable** operation so the per-step cost becomes O(d²) (state-recurrent) instead of O(n·d) (sequence-length dependent). Trades some fidelity for constant-time-per-token inference and a fixed memory footprint. Modern variants (DeltaNet, GLA, Mamba-2, Video-DeltaNet) restore quality via gating and delta-rule state updates, and now show up in hybrid stacks alongside softmax attention.

**Prereqs:** [../fundamentals/attention.md](../fundamentals/attention.md), [multi-head-attention.md](multi-head-attention.md)
**Related:** [mla.md](mla.md), [transformer-block.md](transformer-block.md)

---

## What it is

Standard softmax attention computes `softmax(QKᵀ/√d)V` — an O(n²·d) operation over the full sequence. Kernel-based reformulations approximate softmax with a factorizable feature map φ so that
```
softmax(QKᵀ) V  ≈  φ(Q) · (φ(K)ᵀ V)
```
The right side can be maintained as a running state `S_t = Σ_{i≤t} φ(k_i) v_iᵀ`, updated in constant time per token. Attention then reduces to a matrix-vector product `φ(q_t)ᵀ S_t`.

Delta-rule variants swap the pure accumulation for a **write-with-forgetting** update:
```
S_t = (I - β_t φ(k_t) φ(k_t)ᵀ) · S_{t-1} + β_t φ(k_t) v_tᵀ
```
This lets the state overwrite stale associations rather than only adding to them — a large empirical improvement over classic linear attention.

## How it works

- **Prefill** — process the whole prompt with either the closed-form parallel formulation (chunk-wise) or the recurrent one, depending on kernel availability.
- **Decode** — each new token updates the fixed-size state S and reads a query against it. Memory is constant, throughput is per-token constant.
- **Gating and delta** — learnable per-token gates β_t control write strength, and delta updates prevent unbounded accumulation.

Hybrid stacks (softmax + linear) keep softmax for interactions that need fine-grained pairwise structure (typically cross-modality attention) and use linear branches for long homogeneous streams (long video, long context). Video DeltaNet is one such stack: per-frame linear memory for video-to-video, softmax kept for video ↔ text/audio.

## Why it matters

- **Constant-memory KV.** Long-context and streaming workloads no longer scale KV in sequence length.
- **Throughput.** Decode step becomes an O(d²) matmul with no growing tensor to read from HBM.
- **Practical for video / audio.** Video DeltaNet reports 14.5× speedup over dense softmax on a 14.3-second 768p DiT rendering while matching quality — a regime softmax cannot serve at production latency.

## Gotchas & tricks

- Pure linear attention (no gating) loses accuracy on tasks needing fine-grained recall — always use a delta / gating variant.
- The state matrix S is small (d² floats per layer per head) but *dense* — do not shard it the way you would a growing KV cache.
- Staged teacher-alignment (introduce the linear branch gradually into a pretrained softmax model) is the reliable recipe for retrofits; from-scratch linear-only models are still behind softmax at frontier quality.
- Kernel maps that use `elu(x)+1` or `1+relu(x)` are the safe defaults; polynomial feature maps blow up magnitudes.

## Sources

- "Video DeltaNet: A Video-Native Hybrid Attention for Livestream Video Generation" — Xi et al., 2026 — [arXiv:2609.20744](https://arxiv.org/abs/2609.20744).
- "Transformers are RNNs" — Katharopoulos et al., 2020 — foundational linear-attention paper.
- "Parallelizing Linear Transformers with the Delta Rule over Sequence Length" — Yang et al., 2024 — delta-rule variant.
