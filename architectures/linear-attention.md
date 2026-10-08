# Linear Attention

*Depth — attention computed in `O(N)` time by swapping the softmax for a kernel that decomposes over keys and values.*

**TL;DR:** Standard softmax attention is `O(N²)` because the softmax over the N keys for each of N queries cannot be factored. Linear attention replaces the softmax with a feature map `φ(·)` so that `softmax(QKᵀ)V ≈ φ(Q) (φ(K)ᵀ V)` — the right-hand side has an `O(N·d²)` kernel-sum state that is updated incrementally. Linear attention sacrifices some retrieval-quality on long range to buy sub-quadratic cost; modern uses mix it with softmax (**hybrid attention**) rather than replacing softmax entirely.

**Prereqs:** [attention.md](../fundamentals/attention.md), [multi-head-attention.md](./multi-head-attention.md)
**Related:** [mla.md](./mla.md), [hla.md](./hla.md), [_moe.md](./_moe.md)

---

## What it is

Softmax attention computes

$$
\mathrm{Attn}(Q, K, V) = \mathrm{softmax}(QK^\top / \sqrt{d}) V
$$

for `Q, K, V ∈ ℝ^{N×d}`. The softmax makes the kernel `k(q, k) = exp(q·k/√d)` non-decomposable: you cannot split `exp(q·k)` into `φ(q)·φ(k)` for any finite feature map.

Linear attention replaces the softmax with a kernel that *is* decomposable:

$$
\mathrm{Attn}_{\text{lin}}(Q, K, V) = \frac{\varphi(Q)\,[\varphi(K)^\top V]}{\varphi(Q)\,[\varphi(K)^\top \mathbf{1}]}
$$

Now the inner `φ(K)ᵀ V` is `d × d` and `φ(K)ᵀ 1` is `d × 1`. These are state tensors updated incrementally as you walk the sequence. Decode is `O(d²)` per token, independent of sequence length.

## How it works

### Common feature maps

- `φ(x) = elu(x) + 1` (Katharopoulos 2020, the original "Transformers are RNNs"). Keeps `φ(x) ≥ 0` so the kernel stays non-negative.
- Random Features / Performer: `φ(x) = exp(w·x − ‖x‖²/2)` with random `w`. Unbiased estimator of softmax attention.
- Learned feature maps: `φ` is a small MLP, often with explicit low-rank structure.
- **Gated** linear attention variants add an input-dependent decay term to the running state — this is where SSMs (Mamba) meet linear attention.

### Causal / autoregressive form

For a causal mask, maintain two running sums (one per head, per layer):

```
S_t = Σ_{s ≤ t} φ(k_s) v_sᵀ       (d × d matrix)
z_t = Σ_{s ≤ t} φ(k_s)             (d-vector)
```

Output at step `t`:

```
y_t = (φ(q_t)ᵀ S_t) / (φ(q_t)ᵀ z_t)
```

`S_t` is updated in `O(d²)` per step. No KV cache that grows with sequence length — the "cache" is the fixed-size state tensor.

### Why quality drops

The feature-map kernel is a cheap *approximation* of `exp(q·k)`. For retrieval-heavy tasks (needle-in-haystack, long-range cross-reference) the approximation smears attention mass across many keys, dulling the sharp lookups softmax does well. Benchmarks that force this failure mode (RULER, needles-of-ten) separate linear from softmax attention by large margins.

## Why it matters

- **Compute scales linearly in `N`.** For 1M-token context, that's a 1000× factor over softmax (naive) or 32× over FlashAttention (which still has `O(N²)` compute but with much better constants).
- **Decode is `O(d²)` per token, independent of context length.** No KV cache memory growth. This is the single property that keeps SSMs, Mamba, and GLA families in the game.
- **The foundation for hybrid attention.** Modern designs (Jamba, Griffin, Zamba, HLA) interleave a few softmax layers with many linear/SSM layers. Linear attention handles bulk compute; softmax handles rare precise lookups.

## Gotchas & tricks

- **Pure linear attention loses retrieval**. If you need in-context learning, few-shot or long-range copy, do not go full linear. Keep at least a few softmax layers.
- **Numerics of `φ(K)ᵀ V` matter.** The `d × d` state tensor can accumulate noise. Chunked recomputation (Mamba-2 style) refreshes it periodically.
- **Causal masks require careful kernels.** Causal softmax attention has mature FlashAttention kernels; causal linear attention has fewer, and performance depends heavily on the specific feature map.
- **Hybrid ratio is a design choice.** Static ratios (every Nth layer is softmax) are easy to tune; dynamic routing (as in [hla.md](./hla.md)) lets the model spend softmax where it's needed.
- **Linear-attention expressivity ≤ softmax.** Information-theoretically, a bounded-dimension state cannot store arbitrary associative memory. Expect gaps on retrieval benchmarks even with the best feature maps.

## Sources

- Paper: *Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention* — Katharopoulos et al., 2020.
- Paper: *Rethinking Attention with Performers* — Choromanski et al., 2021 — random-feature kernel for softmax.
- Paper: *Gated Linear Attention Transformers with Hardware-Efficient Training* — Yang et al., 2024.
- Paper: *HLA: Expressive Hybrid Linear Attention via Chunk-Wise Dynamic Mixing* — Chen et al., 2026 — [arXiv:2610.05842](https://arxiv.org/abs/2610.05842).
