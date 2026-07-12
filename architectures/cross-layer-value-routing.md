# Cross-Layer Value Routing (CLVR)
*Depth — cross-depth information sharing for linear-attention stacks at linear cost.*

**TL;DR:** Recurrent linear attention (DeltaNet, Gated DeltaNet, Kimi Delta, Gated DeltaNet-2) keeps a per-layer finite-state memory. Because each layer's recurrent state is independent, information can't flow across depth the way it does through the full-attention KV cache. CLVR routes a small slice of value vectors from lower layers to upper layers via a learned lightweight gate, recovering some of what softmax attention gives for free — while preserving linear-time complexity in sequence length.

**Prereqs:** [attention.md](../fundamentals/attention.md), [multi-head-attention.md](multi-head-attention.md)
**Related:** [_linear-attention.md](_linear-attention.md) · [transformer-block.md](transformer-block.md)

---

## What it is

A per-layer add-on for linear-attention stacks. Each layer $\ell$ exposes a small routing view of its value tensor $V^{(\ell)}$ to a limited window of upper layers. The upper layer's linear-attention state update incorporates the routed values through a learned gate, so information can bypass the per-layer recurrent state without needing global attention over the sequence. Overhead is a small constant per layer; sequence-length scaling stays linear.

## How it works

Given a $L$-layer stack of linear-attention blocks with hidden width $d$:

1. Each layer emits a low-rank projection $\tilde V^{(\ell)} = W^{R,(\ell)} V^{(\ell)}$ of width $d_r \ll d$. This is the *route bus*.
2. Upper layer $\ell'$'s recurrent update reads a gated mix of its own $V^{(\ell')}$ and a windowed aggregate $\sum_{\ell < \ell'} g^{(\ell,\ell')} \tilde V^{(\ell)}$, where $g^{(\ell,\ell')}$ is a learned per-layer-pair gate.
3. The recurrent state update rule of the linear-attention variant (DeltaNet's outer-product update, Kimi Delta's delta-rule, etc.) is unchanged in shape — only the value stream is augmented.

Cost per token: $O(d_r \cdot L_\text{route})$ extra multiplies, where $L_\text{route}$ is the routing window (typically small). Total complexity remains $O(S \cdot d)$ per layer in sequence length.

## Why it matters

- **Small architectural change, measurable gain.** In the paper's 350M / 15B-token training regime, CLVR consistently improves linear-attention baselines at negligible wall-clock cost.
- **Closes a real gap.** Recurrent linear attention's weakness is depth-wise information flow; CLVR is a targeted fix rather than a return to full attention.
- **Composes with hybrid stacks.** Softmax-linear hybrids (which the same paper shows dominate end-task quality per FLOP) benefit from CLVR on the linear layers without changing the softmax layers.

## Gotchas & tricks

- The routing window matters — routing across the whole stack becomes wasteful past a small window; the paper's ablations settle on modest depths.
- $d_r$ is a compression knob; too small kills the gain, too large eats the linear-cost budget.
- Gates need proper initialization — starting near zero avoids destabilizing the linear-attention state update in early training.
- Doesn't rescue pure softmax replacement — CLVR closes a gap, it doesn't erase it. Hybrid softmax-linear stacks still win on end-task quality.

## Sources

- Paper: *Linear Attention Architectures: Mechanisms, Trade-offs, and Cross-Layer Routing* — Cerruti et al., ETH Zurich, 2026 — [arXiv:2607.07953](https://arxiv.org/abs/2607.07953).
- Background: *DeltaNet* — Yang et al., 2024. See [_linear-attention.md](_linear-attention.md).
- Background: Kimi Delta Attention — Moonshot AI, 2025 (per the referenced Kimi tech report).
