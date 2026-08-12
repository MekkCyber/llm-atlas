# Grouped Differential Latent Attention (GDLA)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Fuse **differential attention** (dual-softmax subtractive attention that cancels noise) with **MLA** (Multi-head Latent Attention with compressed KV cache). GDLA runs the differential-attention subtraction over MLA-reconstructed keys and values, keeping both the sharpness benefit of differential attention and MLA's ~50× KV-cache reduction. Introduced by Motif 3 (2026) as the frontier attention primitive for long-context MoEs.

**Prereqs:** [differential-attention.md](differential-attention.md), [mla.md](mla.md), [multi-head-attention.md](multi-head-attention.md)
**Related:** [../case-studies/motif-3.md](../case-studies/motif-3.md), [qk-norm.md](qk-norm.md)

---

## What it is

Differential attention and MLA are pitched as alternatives in the literature — differential improves attention *quality*, MLA improves cache *footprint*. GDLA's claim is that they're actually **complementary**: you can compress the KV cache MLA-style and still do the differential-attention subtraction on the reconstructed representations, without giving up either win.

The naïve combination fails: differential attention needs *two* K matrices per head, and MLA compresses to *one* latent per token. GDLA introduces a **grouped** subtraction — the two "attention halves" share the MLA latent but have separate up-projections and separate learned mixing scalars.

## How it works

Given the MLA latent `c^KV` per token (shape `[d_c]`) and the rotary slice `K^R` (shape `[d_h^R]`):

1. **Per-head, two up-projections instead of one:**
   ```
   K^(1) = W^{UK,(1)} · c^KV        K^(2) = W^{UK,(2)} · c^KV
   V^(1) = W^{UV,(1)} · c^KV        V^(2) = W^{UV,(2)} · c^KV
   Q^(1) = W^{UQ,(1)} · c^Q         Q^(2) = W^{UQ,(2)} · c^Q
   ```
   Each half concatenates with its own rotary slice, same as MLA.

2. **Two attention maps per head, computed on the same cached latent:**
   ```
   A_i^(1) = softmax( Q^(1) · (K^(1))^T / √d )
   A_i^(2) = softmax( Q^(2) · (K^(2))^T / √d )
   ```

3. **Grouped subtraction with a learned scalar `λ`:**
   ```
   head_out = ( A^(1) − λ · A^(2) ) · V^(mix)
   ```
   where `V^(mix)` is either a learned interpolation of `V^(1)` and `V^(2)` or the "positive" branch alone (Motif 3's exact choice is one of the paper's engineering knobs).

4. **KV cache unchanged from MLA.** Because both attention halves derive from the same `c^KV`, the cache stores one latent per token per layer plus the shared rotary slice — same footprint as vanilla MLA.

The "**grouped**" name refers to the shared-latent, per-head-doubled-projection structure: within a head, the two halves share the same *group* of cached latents but project them differently.

## Why it matters

- **Doubles the "wins" of the modern attention stack.** MLA alone reduces cache; differential alone sharpens attention; GDLA does both in one operator.
- **No cache penalty vs MLA.** The two attention halves live in the up-projections (weights, not cache). KV-cache footprint stays at MLA's `d_c + d_h^R` per token per layer.
- **Better long-context recall than either primitive alone.** Motif 3's tech report positions GDLA as a key contributor to the model's long-context and reasoning results.
- **Sets the pattern for future frontier MoEs.** If GDLA holds up in reproductions, expect it to displace vanilla MLA as the default long-context attention primitive.

## Gotchas & tricks

- **Compute doubles per head.** Two softmax + matmul per head vs one for MLA. Wall-clock impact is smaller because the MLA-absorbed-projection trick still applies to both halves.
- **λ initialization matters more than in vanilla differential attention.** Because both halves share the latent, λ near 1.0 risks canceling most of the signal. Init lower (0.3–0.5) and let it grow.
- **The two up-projections must be initialized differently.** If `W^{UK,(1)} = W^{UK,(2)}` at init, `A^(1) = A^(2)` and the subtraction produces zero — the model can't learn out of that fixed point. Standard practice: random init with different seeds.
- **Kernel is more complex than MLA's.** MLA already needs a custom attention kernel; GDLA needs one that computes two attention maps against the same cached latent. The paper documents this as one of the engineering costs.
- **Doesn't compose trivially with GQA/MQA.** Both alternative sharing schemes assume one K per head; GDLA's two-K structure needs adaptation.
- **QK-norm interacts.** As with vanilla differential attention, over-sharpening is a risk when stacking with QK-norm; tune both together.

## Sources

- Paper: *Motif 3 Technical Report* — Motif Technologies, 2026 — the source paper introducing GDLA.
- Background: *Differential Transformer* — Microsoft Research, 2024 — differential attention primitive.
- Background: *DeepSeek-V2* — DeepSeek, 2024 — MLA primitive.
