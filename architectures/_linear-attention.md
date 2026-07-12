# Linear Attention

*Taxonomy — recurrent-memory attention variants that scale linearly in sequence length.*

**TL;DR:** Softmax attention is $O(S^2)$ in sequence length. Linear-attention variants replace softmax with a rule that lets attention be computed by a *recurrent finite-state update* — $O(S \cdot d)$ per layer. Modern recurrent linear-attention variants (DeltaNet, Gated DeltaNet, Kimi Delta Attention, Gated DeltaNet-2) close much of the quality gap to softmax at 350M / 15B-token scale; hybrid softmax-linear stacks win end-task quality per FLOP. **Cross-Layer Value Routing (CLVR)** further tightens the gap with a cheap depth-wise information channel.

**Related taxonomies:** *(none yet)*
**Depth files covered here:** [cross-layer-value-routing](cross-layer-value-routing.md)

---

## The problem

Full softmax attention over a sequence of length $S$ costs $O(S^2 \cdot d)$ per layer. Long-context inference (32k, 128k, 1M) makes this prohibitive both in FLOPs and KV-cache memory. Linear attention aims to be a drop-in replacement that reads and writes a *finite-state* memory per token — total $O(S \cdot d)$ — while retaining softmax-competitive quality.

## The shared pattern

All modern recurrent linear-attention variants can be written in a unified recurrent-memory form:

$$
S_t = f(S_{t-1},\ k_t,\ v_t) \qquad y_t = g(q_t,\ S_t)
$$

where $S_t \in \mathbb{R}^{d \times d}$ (or a lower-rank compression) is the recurrent state, $f$ is a write rule (often an outer-product update possibly modulated by a data-dependent gate), and $g$ is a read rule (usually $q_t^\top S_t$). Variants differ on:

1. **Update rule** — pure outer-product (Linear Transformer), delta rule (DeltaNet), gated delta rule (Gated DeltaNet), sign-corrected delta (Kimi Delta).
2. **Gating** — no gating, per-token gate, per-dimension gate.
3. **Depth-wise information flow** — none by default; CLVR adds it as a cross-layer bus.

## Variants

| Technique | Key idea | Main tradeoff | When it wins |
| --- | --- | --- | --- |
| Linear Transformer (no depth file yet) | Kernelize softmax as feature-map dot product | Quality lags softmax on long-range retrieval | Historical baseline |
| DeltaNet (no depth file yet) | Delta-rule state update replaces MSE outer product | Complex state; needs care in kernel | Strongest ungated recurrent baseline |
| Gated DeltaNet (no depth file yet) | Adds data-dependent gate on the state update | More params, better sequence-length transfer | Best pure linear at modest scales |
| Kimi Delta Attention (no depth file yet) | Sign-corrected delta update from Kimi's stack | Proprietary tuning details | Kimi k-series production |
| Gated DeltaNet-2 (no depth file yet) | Refined gating + tighter state normalization | Slightly higher constant cost | Current recurrent-linear frontier |
| Hybrid softmax-linear (no depth file yet) | Stack a few softmax layers among many linear layers | Adds back $O(S^2)$ where placed | **Wins end-task quality / FLOP** in the paper's sweep |
| [CLVR](cross-layer-value-routing.md) | Depth-wise value bus with learned gates | Small constant overhead | Recovers cross-layer info flow at linear cost |

## How to choose

**For a new stack**, gated linear-attention (Gated DeltaNet-2 class) with a few interleaved softmax layers is the current default — the hybrid closes the softmax gap while keeping the majority of layers at linear cost. **CLVR** adds a cheap depth-wise channel that helps both pure-linear and hybrid stacks. **Pure linear attention** is only worth it when memory / throughput at extreme context is the hard constraint.

## Adjacent but distinct

- **Sliding-window / sparse attention.** Also sub-quadratic but by attending to a subset of positions rather than by finite-state recurrence.
- **SSM / Mamba.** State-space models share the "finite-state recurrent update" framing but derive it from continuous-time systems rather than attention approximation.
- **MLA / GQA.** KV-cache compression, not sequence-length complexity — orthogonal axis.

## Sources

- Paper: *Linear Attention Architectures: Mechanisms, Trade-offs, and Cross-Layer Routing* — Cerruti et al., ETH Zurich, 2026 — [arXiv:2607.07953](https://arxiv.org/abs/2607.07953).
- Paper: *Transformers Are RNNs* — Katharopoulos et al., 2020 — original linear attention.
- Paper: *DeltaNet* — Yang et al., 2024.
- Paper: *Gated Linear Attention (GLA / Gated DeltaNet)* — Yang et al., 2024.
- Model: Kimi Delta Attention — Moonshot AI, referenced from the Kimi k-series stack.
