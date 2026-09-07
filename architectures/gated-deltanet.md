# Gated DeltaNet
*Depth — a linear-attention recurrent block whose fixed-size state summarizes an entire context.*

**TL;DR:** Gated DeltaNet (GDN) replaces softmax attention's O(L²) full-context lookup with a bounded recurrent state updated by a **delta rule** — each new key/value pair overwrites the state's component along the current key direction, gated by learned decay and write-strength signals. In hybrid architectures like Qwen3.8-27B (48 GDN layers + 16 softmax-attention layers), GDN carries most of the long-range memory while attention layers handle sharp retrieval.

**Prereqs:** [attention](../fundamentals/attention.md), [multi-head-attention](../architectures/multi-head-attention.md)
**Related:** [mla](mla.md), [_moe](_moe.md)

---

## What it is

A linear-attention variant that maintains a matrix-valued state $S \in \mathbb{R}^{d_k \times d_v}$ per head and per sequence, updated recurrently:

- Read: $y_t = q_t \, S_{t-1}$
- Write: $S_t = \alpha_t \odot S_{t-1} + \beta_t \, k_t v_t^\top - \gamma_t (k_t k_t^\top) S_{t-1}$

Where $\alpha_t$ is a decay gate, $\beta_t$ a write-strength gate, and $\gamma_t$ the delta term that overwrites the state's component along $k_t$'s direction. State size is independent of sequence length, giving O(L) memory and O(L) attention compute — the "linear" in linear attention. Hybrid architectures pair GDN with a few softmax-attention layers to recover the exact retrieval capability GDN's compressed state loses.

## How it works

Reading the delta rule intuitively: for each new token, you inspect $S \cdot k_t$ (what the state currently says about the key direction $k_t$), subtract that, and add the new $k_t v_t^\top$ pair scaled by the write-strength gate. Any prior write with a similar key direction is *overwritten* rather than accumulated on top of. The decay gate $\alpha_t$ acts as a leak on the whole state.

Parametrizations that matter for quantization (see [nvfp4](../quantization/nvfp4.md)):
- Decay gate: sigmoid / exponential — bounded in $[0, 1]$.
- Write-strength gate: softplus / exponential — non-negative.

Because both gates pass through smooth non-linearities that compress input range, an 11% GEMM error in the gate projection compresses to ~2% output error — the "fragile-looking" gates are actually the most quantization-robust components in the block.

## Why it matters

- **Long context at bounded memory.** State size is a hyperparameter, not the sequence length. GDN handles 32K–64K contexts without KV-cache blowup.
- **Quantization-friendly.** The delta rule *bounds* injected quantization noise: each write overwrites along the current key direction, so noise from prior tokens is forgotten within hundreds of steps — far faster than the decay-gate horizons — instead of compounding along the recurrence.
- **Complements attention.** Hybrids pair GDN's compressed continuous memory with softmax attention's precise-retrieval spikes. Empirically 3:1 GDN:attention ratios (as in Qwen3.8-27B) work well.

## Gotchas & tricks

- **Not a free replacement for attention.** Pure-GDN models lose sharp long-range retrieval; keep some attention layers.
- **Quantize aggressively.** Contrary to intuition, GDN is the *easy* half of a hybrid to quantize — see NVFP4 W4A4 results in Minima.
- **State initialization.** Zero-initialize; treat the first ~hundreds of tokens as a warm-up during which read outputs are uninformed.
- **Batching.** Per-sample recurrent state means each sequence in a batch needs its own state; take care in ragged-batch serving.
- **Kernel efficiency.** Naïve implementation is slow; production uses fused kernels (analogous to FlashAttention for softmax attention).

## Sources

- Paper: *Why Gated DeltaNet Survives 4-Bit Quantization: NVFP4 W4A4 for the Recurrent Half of a Hybrid 27B LLM* — Kozyrev & Maiboroda, Minima-AI, 2026 — the quantization-mechanism study — https://arxiv.org/abs/2609.04098
- Paper: *Gated Linear Attention Transformers with Hardware-Efficient Training* — Yang et al., 2024 — gated linear-attention family.
- Paper: *Parallelizing Linear Transformers with the Delta Rule over Sequence Length* — Yang et al., 2024 — delta-rule linear attention.
- Model card: `Qwen/Qwen3.8-27B` — the reference hybrid using 48 GDN + 16 softmax-attention layers.
