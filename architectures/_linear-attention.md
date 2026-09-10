# Linear attention

*Taxonomy — attention variants with O(T) time and O(1) memory decoding.*

**TL;DR:** Softmax attention is O(T²) time and O(T) memory during decoding; linear attention replaces the softmax with a kernel so the state can be maintained as a fixed-size recurrent memory. All variants share the same skeleton — a matrix associative memory, a write rule, and a read rule — and differ in how they modulate writes to avoid rapidly forgetting or blindly overwriting.

**Related taxonomies:** [_moe](_moe.md), [_normalization](_normalization.md)
**Depth files covered here:** [kalman-delta-network](kalman-delta-network.md)

---

## The problem

Softmax attention conditions on all past K/V explicitly. That gives arbitrary retrieval but costs O(T) KV cache and O(T) FLOPs per decode step. For long context (100K+ tokens) and constant-latency streaming, both costs become prohibitive.

## The shared pattern

Every variant is a **matrix-valued RNN** over a fixed-size state `S_t ∈ ℝ^{d_k × d_v}`:

```
S_t = f(S_{t−1}, k_t, v_t)   # write rule (state update)
o_t = q_tᵀ S_t               # read rule (query)
```

Design choices:

- **Write rule** — pure additive (`S += k vᵀ`), delta-rule (overwrite associations for the current key), gated (input/forget gate over the state), uncertainty-aware (Kalman-style).
- **Kernel on q, k** — identity (Linear Transformer), elementwise ReLU / ELU+1, feature maps for softmax approximation.
- **Scan** — most rules admit an associative scan with O(log T) parallel depth on GPU.

## Variants

| Technique | Key idea | Main tradeoff | When it wins |
| --- | --- | --- | --- |
| Linear Transformer (Katharopoulos+ 2020) | Kernelised softmax; pure additive writes | Rapid state saturation — forgets old associations | Baseline / theory |
| RetNet (Sun+ 2023) | Exponential decay on the additive rule | Fixed decay rate is not adaptive | Any-length inference at fixed cost |
| Mamba / S6 (Gu+ 2023) | Selective state-space with input-dependent Δ | Discrete-time SSM parametrisation is subtle to tune | Long-range language modeling |
| DeltaNet (Schlag+ 2021) | Delta-rule overwrite: replace old value for the current key | Overwrite strength is a scalar per token | Associative recall tasks |
| Gated DeltaNet (Yang+ 2024) | Adds a forget gate to delta rule | Two gates to tune | Improved long-context recall |
| [kalman-delta-network](kalman-delta-network.md) | Kalman filter — write weight depends on tracked covariance | Adds diagonal or isotropic uncertainty state | When quality gains at fixed scan cost matter |
| Titans (Behrouz+ 2024) | Learned long-term + short-term memory hierarchy | Two-memory bookkeeping | Very long context with retrieval needs |

## How to choose

- **Default long-context backbone today:** Mamba-2 or a hybrid (a few softmax-attention layers interleaved) — hybrids consistently outperform pure linear-attention on associative recall.
- **Retrieval-heavy workloads:** delta-rule family (DeltaNet, Gated DeltaNet, KDN) — pure additive/gated rules forget too fast.
- **Quality-sensitive research at fixed scan cost:** KDN — uncertainty-aware writes are a small addition on top of delta-rule and dominate at ≤1.3B.
- **Do not go pure linear** for reasoning benchmarks yet; interleave with softmax layers.

## Adjacent but distinct

- **Sliding-window attention** — still softmax, just bounded context. Constant memory but no explicit state.
- **MLA / GQA** — reduce K/V head count but stay softmax. See [../architectures/mla.md](mla.md).
- **RWKV** — RNN-shaped model with attention-like reads; often grouped with linear-attention but has its own family history.

## Sources

- Katharopoulos et al., *Transformers are RNNs* — 2020 — https://arxiv.org/abs/2006.16236
- Sun et al., *Retentive Network* — 2023 — https://arxiv.org/abs/2307.08621
- Gu, Dao — *Mamba: Linear-Time Sequence Modeling with Selective State Spaces* — 2023 — https://arxiv.org/abs/2312.00752
- Yang et al., *Gated Linear Attention* — 2023 — https://arxiv.org/abs/2312.06635
- Bui, Huang, Ying — *Kalman Delta Networks* — 2026 — https://arxiv.org/abs/2609.07816
