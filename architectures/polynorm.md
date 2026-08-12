# Expert-Specific PolyNorm
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A polynomial-normalization activation — activations pass through `x → σ(x) · P(x)` where `P` is a learnable low-degree polynomial — with per-expert weights inside an MoE layer. Each expert learns its own polynomial coefficients, letting the activation function itself specialize alongside the expert weights. Introduced (in this expert-specific form) by Motif 3 (2026) as one of the enablers of stable 384-expert fine-grained MoE routing.

**Prereqs:** [_moe.md](_moe.md), [transformer-block.md](transformer-block.md), [../fundamentals/attention.md](../fundamentals/attention.md)
**Related:** [../case-studies/motif-3.md](../case-studies/motif-3.md), [deepseek-moe.md](deepseek-moe.md)

---

## What it is

MoE experts typically share a single activation function (SwiGLU, GeGLU, ReLU). As expert count grows fine-grained (256 → 384 in Motif 3), each expert has less capacity, and forcing them all through the same nonlinearity may bottleneck specialization. Expert-specific PolyNorm lets each expert learn its own low-degree polynomial component of the activation, giving the nonlinearity itself a per-expert degree of freedom.

"PolyNorm" refers to a small family of polynomial activation functions used as a normalization / activation hybrid; Motif 3's contribution is making it **expert-specific inside an MoE FFN**.

## How it works

A generic PolyNorm activation with degree `k`:
```
PolyNorm(x) = γ · (Σ_{i=0..k} a_i · x^i)   /  √( Σ_{i=0..k} a_i² · E[x^{2i}] )
```

The polynomial `P(x) = Σ a_i x^i` provides the shape (learnable `a_0…a_k`); the denominator normalizes the output variance so the activation stays well-scaled regardless of coefficient values. `γ` is a learned per-channel scale.

**Expert-specific version.** In an MoE layer with `N` routed experts, maintain a separate coefficient vector `[a_0, …, a_k]_i` for each expert `i`. During routing, when expert `i` fires on token `t`:
```
h_i(t) = W_up,i · PolyNorm_i( W_gate,i · t )
h(t) = Σ_{i ∈ topK(t)} g_i(t) · W_down,i · h_i(t)
```

Coefficient storage per layer: `N × (k+1)` scalars. At `N = 384` and `k = 3` that's ~1.5K scalars per layer — negligible vs the expert weight budget.

## Why it matters

- **Fine-grained MoE keeps scaling.** DeepSeek-V3 pushed to 256 experts with shared activations; Motif 3's 384-expert regime is where per-expert activation flexibility starts paying off — small experts need every degree of freedom.
- **Polynomial ≠ piecewise-linear.** PolyNorm's smoothness helps optimization stability in a way ReLU/SwiGLU don't, especially at low expert capacity where non-smooth activations amplify router-choice discontinuities.
- **Cheap.** Coefficient count is a rounding error; the extra ops per activation are also small (a few multiply-adds vs one SwiGLU call).
- **Normalization built in.** The `√E[x^{2i}]` denominator keeps activations well-scaled without needing a separate LayerNorm inside the expert.

## Gotchas & tricks

- **Degree `k` is a lever.** `k = 2` is under-expressive; `k ≥ 4` starts introducing numerical instabilities (small coefficient perturbations blow up high-order terms). Motif 3's exact degree is one of the paper's implementation choices.
- **Coefficient initialization.** Init each expert's `a_i` near a common target (e.g. approximate the identity or SwiGLU) then let expert-specific training peel them apart. Random-init coefficients per expert makes training unstable.
- **Normalization must use running statistics, not batch statistics.** Per-batch normalization by `E[x^{2i}]` couples experts across the batch and defeats the point of expert-specific activations.
- **Interacts with load-balancing.** If one expert's polynomial makes it especially "friendly" to a category of tokens, the router will over-route to it. Standard load-balancing (aux-loss-free) handles this, but the coupling is real.
- **Not a drop-in for dense models.** In a single-FFN dense model, per-expert polynomials degenerate to a single shared activation — nothing gained. This technique is MoE-native.

## Sources

- Paper: *Motif 3 Technical Report* — Motif Technologies, 2026 — the source paper for the expert-specific formulation.
- Background: PolyNorm activation family — prior work on polynomial normalization activations in transformers.
