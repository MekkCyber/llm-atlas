# Rotary Position Embedding (RoPE)
*Depth — the rotary position encoding that is now the default in modern LLMs.*

**TL;DR:** Instead of adding a position signal to the token embedding (sinusoidal), **rotate** the query and key vectors by a position-dependent angle before the attention dot product. Pairs of dimensions `(2i-1, 2i)` are rotated by `m·θ_i`, with `θ_i = base^(-2i/d)` and `base = 10000`. Because rotating q by `mθ` and k by `nθ` leaves the dot product as a function of `(n-m)θ`, **relative position is encoded implicitly** — you keep the absolute API (one function per token) but get relative semantics for free. No parameters added. Plays well with Flash Attention, linear attention, GQA/MQA. Extensible after training via position interpolation, NTK-aware scaling, YaRN, and base-frequency scaling — which is why every modern LLM (Llama, Qwen, DeepSeek, Mistral, Gemma, PaLM, Kimi) uses it.

**Prereqs:** [attention](attention.md), [sinusoidal-encoding](sinusoidal-encoding.md)
**Related:** [_positional-encoding](_positional-encoding.md) · [multi-head-attention](../architectures/multi-head-attention.md)

---

## What it is

Positional encoding with three structural choices:

1. **What is injected** — a rotation applied to q and k vectors.
2. **Where it enters** — after the q/k projections, before the attention dot product. **Every layer**, not just once at the input.
3. **Parameterization** — fixed, no learned parameters. Frequencies are geometrically spaced powers of a base.

Applied only to `Q` and `K`. `V` is **not rotated**. The rotation is orthogonal, so it preserves norms — no activation drift.

RoPE is the solution that the field converged on (RoFormer 2021 → GPT-NeoX / PaLM 2022 → Llama family 2023+) because it combines the *absolute* API of sinusoidal with the *relative* semantics of learned-bias schemes, without adding parameters and without breaking modern attention kernels.

---

## How it works

### The 2D case — one pair of dimensions

Consider a single pair of coordinates `(x_1, x_2)` of a query at position `m`. RoPE applies a 2D rotation by angle `mθ`:

```
[ x_1' ]   [  cos(mθ)   -sin(mθ) ]   [ x_1 ]
[      ] = [                     ] · [     ]
[ x_2' ]   [  sin(mθ)    cos(mθ) ]   [ x_2 ]
```

Identically for keys at position `n`, with angle `nθ`.

Why angle `= m · θ`? The paper (RoFormer, Sec. 3.4.1) derives it from the requirement that the attention inner product depends only on `(n - m)`. Starting from `⟨f_q(x, m), f_k(y, n)⟩ = g(x, y, n-m)` and fixing the radial part, the angular function `ϕ(m)` must be an arithmetic progression: `ϕ(m) = mθ`. Linearity in `m` is what makes the relative-position property work — rotating by `mθ` on q and `nθ` on k composes to a rotation by `(n-m)θ` in the dot product.

### The full-d case — block-diagonal rotation

For a d-dimensional vector (d even), pair up dimensions as `(1,2), (3,4), …, (d-1, d)`. Each pair `i = 1, …, d/2` gets its own frequency `θ_i` and rotates by `m · θ_i`. The full rotation is block-diagonal:

```
         ┌ cos(mθ_1)  -sin(mθ_1)        0                                   ┐
         │ sin(mθ_1)   cos(mθ_1)        0                                   │
         │                        cos(mθ_2)  -sin(mθ_2)                     │
R(m)  =  │                        sin(mθ_2)   cos(mθ_2)                     │
         │                                                  ⋱               │
         │                                           cos(mθ_{d/2}) -sin(...)│
         └                                           sin(mθ_{d/2})  cos(...)┘
```

Sparse, orthogonal, norm-preserving. Applied to q:

```
q_m_rotated = R(m) · W_q · x_m
```

And to k:

```
k_n_rotated = R(n) · W_k · x_n
```

### The frequency spectrum — `base = 10000`

RoPE uses the same geometric frequency spacing as sinusoidal encoding:

```
θ_i = base^(-2(i-1)/d)    for i = 1, …, d/2
base = 10000              (original RoFormer choice; follows Vaswani 2017)
```

So `θ_1 = 1` (fastest rotation, shortest wavelength) and `θ_{d/2} ≈ 10000^{-(d-2)/d} ≈ 1/10000` (slowest rotation, longest wavelength). Low-index dimension pairs carry **fine-grained position**; high-index pairs carry **coarse position** across the whole sequence.

### The relative-position identity

Because `R(m)` is orthogonal, `R(m)^T R(n) = R(n - m)`. So the attention inner product becomes:

```
q_m_rotated^T · k_n_rotated
  = (R(m) W_q x_m)^T (R(n) W_k x_n)
  = x_m^T W_q^T · R(m)^T R(n) · W_k x_n
  = x_m^T W_q^T · R(n - m)  · W_k x_n
```

The absolute positions `m` and `n` dropped out. The attention score depends on `(x_m, x_n)` and the **gap `(n - m)`** — never on `m` and `n` individually. This is the whole point.

### Efficient implementation

Materializing the `d × d` block-diagonal matrix is wasteful. The standard trick: precompute two d-dim vectors `cos_m` and `sin_m` (each `θ_i` repeated for both coordinates of its pair), then:

```
rotated(x) = x * cos_m + rotate_half(x) * sin_m
```

where `rotate_half` swaps each pair `[x_{2i-1}, x_{2i}] → [-x_{2i}, x_{2i-1}]`. Two element-wise multiplies plus an add per token, per layer. `O(d)` per token.

ww

### Long-term decay

RoPE's frequency spectrum gives a natural locality bias: the magnitude of the rotated dot product empirically decays with `|n - m|` (RoFormer Fig. 2). Tokens further apart contribute less on average. The decay is **oscillatory, not monotone** — the paper doesn't claim a clean monotonic bias, and Sec. 4.5.5 acknowledges that the decay argument alone doesn't fully explain RoPE's long-context competence. But it's enough that the model doesn't need an explicit relative-bias term to learn locality.

---

## Extending context length after training

The property the field actually relies on, and where most of the interesting engineering lives. A model trained at context length `L` can be extended to `L' » L` via four main methods:

### 1. Position Interpolation (PI) — Chen et al. 2023 (arXiv 2306.15595)

Divide positions by the scale factor `s = L' / L` before applying RoPE:

```
f'(x, m) = f(x, m · L / L')
```

The maximum angle the rotation formula ever sees stays at `L`. Equivalent to "squishing" the training range to cover the test range. Needs a short fine-tune (~1000 steps for Llama 7B–65B extended 2k → 32k). Frequencies unchanged. Proven by a Lipschitz-smoothness bound (Theorem 2.1): interpolating between trained integer positions is well-bounded, whereas direct extrapolation is not (perplexity explodes past `L`).

### 2. NTK-aware scaling — bloc97 (Reddit, 2023)

Instead of scaling positions, **scale the base frequency**:

```
b' = b · s^(d/(d - 2))
θ'_i = b'^(-2i/d)
```

Positions stay integer. Only the frequency spectrum changes — the lowest frequency gets scaled by roughly `s`, the highest stays ~unchanged. Motivation: PI squishes every frequency uniformly, killing high-frequency content the model relies on; NTK-aware preserves the high-frequency dimensions and only rescales the low ones. First published on Reddit, formalized later in the YaRN paper (Appendix A.2).

### 3. YaRN — Peng et al. 2023 (arXiv 2309.00071)

Best-of-both: a wavelength-aware "NTK-by-parts" that PI-interpolates only the long-wavelength dimensions (which behave as absolute position indicators inside the training window) and leaves the short-wavelength dimensions unchanged. Adds an **attention temperature**:

```
attention = softmax(q^T k / (t · √d)),  with √(1/t) = 0.1 · ln(s) + 1
```

absorbed into the cos/sin tables so runtime is unchanged. Matches PI with **10× fewer tokens, 2.5× fewer training steps**; compatible with Flash Attention.

### 4. Base-frequency scaling (ABF)

Simplest: pick a much larger base. Code Llama used `base = 1,000,000` (2023). Llama 3 uses `base = 500,000` (2024). Kimi k1.5 uses `base = 1,000,000` during long-context activation (see the [Kimi k1.5 case study](../case-studies/kimi-k1-5.md)). Mathematically equivalent to NTK-aware scaling with a specific `s`. Popular because it's a one-line change to the model config.

Which one to use: PI or YaRN with a short adapter fine-tune for best quality; ABF / NTK-aware for zero- or minimal-fine-tune extension. YaRN is the current state-of-the-art for large context extension (e.g., 8k → 128k).

---

## Why it matters

- **Relative out of absolute, no parameters added.** The inner product depends only on `(n - m)`. You don't need a learned bias table per relative distance (T5) or a hand-chosen distance decay (ALiBi).
- **Composes with everything.** RoPE is a modification of Q and K *before* the attention operator, not a term added *inside* it. Flash Attention, linear attention (Performer), sliding-window attention, GQA, MQA, MLA — all work with RoPE unchanged. ALiBi's additive bias is harder to fuse into some kernels.
- **Norm-preserving.** Orthogonal rotation → activation magnitudes don't drift. Additive sinusoidal perturbs norms; learned absolute can drift during training.
- **Applied at every layer.** Sinusoidal is added once at input; its signal dilutes across layers' residual streams. RoPE is re-applied in each attention block, giving a persistent position signal.
- **Extensible after pretraining.** The decisive practical point. PI, NTK-aware, YaRN, and ABF all let you take a trained RoPE model and extend its context length 4× – 32× with little to no fine-tuning. Sinusoidal and learned absolute have no equivalent toolkit. ALiBi extrapolates natively but underperforms RoPE at a fixed context and has no comparable post-hoc extension story.
- **Ecosystem lock-in.** Every open-weights LLM lineage — GPT-NeoX (first at scale), PaLM, Llama 1/2/3, Mistral, Mixtral, Qwen, DeepSeek, Gemma, Kimi — uses RoPE. All tooling (serving, speculative decoding, quantization, context-extension) assumes RoPE.

---

## Gotchas & tricks

- **Applied to Q and K only, not V.** Rotating V breaks the relative-position identity — you'd multiply the attention score by one rotation and the value by another. V is projected and aggregated as usual.
- **Interleaved-pair vs halves layout.** The paper uses interleaved pairs `(2i-1, 2i)`. HuggingFace Llama uses halves `(i, i + d/2)`. A permutation connects them; weights don't transplant across layouts without reshape. When debugging cos/sin tables, verify the layout matches the `rotate_half` implementation.
- **`base` is load-bearing.** Increasing the base from 10000 to 500k–1M during long-context extension is functionally an NTK-aware scaling. Reducing the base shortens the effective context. Don't change it mid-training without a plan.
- **Frequencies are geometric, not linear.** The lowest and highest `θ_i` differ by a factor of `base`. A handful of low-index dimensions rotate many times across the sequence (aliasing possible); a handful of high-index dimensions barely rotate at all (near-constant for the whole sequence).
- **Per-head, not per-layer.** `R(m)` is applied to the head-dim components of q and k. Different heads share the same rotation (the same cos/sin tables); MQA/GQA reduce the number of K/V projections but every head still rotates with the same position-indexed table.
- **FP16 / BF16 precision on cos/sin.** Precomputed tables at FP16/BF16 are adequate for contexts up to ~32k, but longer contexts can lose precision in the sin/cos values of low-frequency dimensions. Some long-context implementations keep the tables in FP32.
- **Don't combine with sinusoidal.** Adding sinusoidal encoding on top of RoPE doubles the position signal and confuses the model. If you use RoPE, don't also add positional encoding at the input.
- **Extrapolation without adaptation is bad.** A RoPE model trained at 2k tested at 8k *without* PI/YaRN/ABF typically explodes — perplexity > 1000 (PI paper Fig. 2). The model has never seen the rotation angles `mθ` for `m > 2k` at inference and doesn't know what to do with them.
- **The decay argument is weak.** RoPE's long-term decay is oscillatory and not monotone. The claim "RoPE has a natural locality bias" is empirically true but theoretically loose — don't rely on the decay for hard guarantees.

---

## Sources

- Paper: *RoFormer: Enhanced Transformer with Rotary Position Embedding* — Su et al., 2021, arXiv 2104.09864 — the original RoPE paper.
- Paper: *Extending Context Window of Large Language Models via Positional Interpolation* — Chen et al., 2023, arXiv 2306.15595 — PI.
- Paper: *YaRN: Efficient Context Window Extension of Large Language Models* — Peng et al., 2023, arXiv 2309.00071 — YaRN and formal write-up of NTK-aware scaling.
- Reddit: bloc97, 2023 — original NTK-aware scaling proposal.
- Paper: *Code Llama: Open Foundation Models for Code* — Rozière et al., 2023, arXiv 2308.12950 — base = 1,000,000 (ABF).
- Paper: *The Llama 3 Herd of Models* — Meta, 2024, arXiv 2407.21783 — RoPE base = 500,000 in Llama 3.
- Paper: *Scaling Laws of RoPE-based Extrapolation* — Liu et al., 2023, arXiv 2310.05209 — base-frequency scaling analysis.
- Paper: *Kimi k1.5: Scaling Reinforcement Learning with LLMs* — Moonshot AI, 2025 — long-context activation uses RoPE base = 1,000,000.
