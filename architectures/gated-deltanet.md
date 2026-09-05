# Gated DeltaNet (GDN)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Gated DeltaNet is a **linear-attention block with a delta-rule state update and per-token forgetting gate**. It maintains a fixed-size recurrent state that summarizes the context, so per-token compute and memory are O(1) in sequence length instead of O(n). Modern hybrid LLMs pair GDN layers with softmax-attention layers to get long-context throughput without giving up in-context retrieval. GDN's forgetting gate is what makes it exceptionally quantization-tolerant — the recurrent state exponentially damps injected noise.

**Prereqs:** [attention](../fundamentals/attention.md), [multi-head-attention](multi-head-attention.md)
**Related:** [../quantization/nvfp4.md](../quantization/nvfp4.md), [transformer-block](transformer-block.md)

---

## What it is

Softmax attention keeps the whole KV cache and looks over it every step — that's the O(n) per-token cost. Linear attention replaces the softmax with a kernel factorization so the "context" collapses into a fixed-size matrix state $S_t \in \mathbb{R}^{d \times d}$ updated recursively.

Gated DeltaNet adds two ingredients on top of vanilla linear attention:

- **Delta rule.** Each new key–value pair *overwrites* the state's response to that key rather than merely adding to it. This gives the state associative-recall behavior instead of the pure lossy averaging of vanilla linear attention.
- **Per-token forgetting gate.** A learned scalar $\alpha_t \in (0, 1)$ multiplies the state before the delta update, exponentially decaying old contents.

The result is a recurrent block with the parallel-scan-friendly structure of linear attention, associative-recall from the delta rule, and controllable memory horizon from the gate.

## How it works

For query $q_t$, key $k_t$, value $v_t$, gate $\alpha_t$, the state update is:

$$
S_t = \alpha_t \cdot \left( S_{t-1} - (S_{t-1} k_t) k_t^\top \right) + v_t k_t^\top
$$

$$
o_t = S_t q_t
$$

- The $(S_{t-1} k_t) k_t^\top$ term is the delta rule: subtract the state's current response to $k_t$ before writing the new value.
- The $\alpha_t \cdot$ prefix is the forgetting gate — a small $\alpha_t$ drops old memory quickly, a large one preserves it.

Modern implementations compute this with a chunkwise parallel scan so training throughput is close to that of softmax attention despite the recurrence.

**In hybrid LLMs**, some layers stay softmax attention (for in-context retrieval on the current turn) and other layers are GDN (for the long-history summary). The mix ratio is a design knob — recent 27B hybrids use roughly 1:1 by count, with GDN making up "the recurrent half."

## Why it matters

- **Constant memory per token.** No KV cache growth over long context. The full state is $O(d^2)$ regardless of sequence length.
- **Quantization tolerance.** The gate's exponential forgetting means quantization noise injected into the state decays out over subsequent steps. NVFP4 W4A4 on a 27B hybrid preserves quality *specifically because* GDN layers absorb activation-quantization noise. Softmax-attention-only architectures don't have this property.
- **Retrieval-competent linear attention.** The delta rule closes most of the "linear attention can't do in-context lookup" gap with softmax attention.

## Gotchas & tricks

- **Not equivalent to Mamba/S4.** GDN is a linear-attention block; Mamba is a selective SSM. Different math, similar goals, comparable results — but the parallel-scan kernels are format-incompatible.
- **Gate initialization is load-bearing.** Init $\alpha_t$ too small and the model forgets everything at start of training; too large and it can't gate off distracting content.
- **Hybrid layouts matter more than the linear block choice.** Where you put GDN layers (early / late / interleaved) affects downstream retrieval more than which linear-attention variant you pick.
- **Pure GDN loses long-context needle-in-haystack.** The recurrent state can't hold arbitrary point-lookups; keep some softmax layers for the retrieval-heavy passes.

## Sources

- Paper: *Gated Delta Networks: Improving Mamba2 with Delta Rule* — Yang et al., 2024, arXiv:2412.06464 — original GDN.
- Paper: *Why Gated DeltaNet Survives 4-Bit Quantization* — Kozyrev & Maiboroda, 2026 — [arXiv:2609.04098](https://arxiv.org/abs/2609.04098) — quantization analysis of GDN in a hybrid 27B LLM.
- Related: DeltaNet (Schlag et al., 2021), Mamba/Mamba2 (Gu & Dao, 2023–2024).
