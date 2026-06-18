# Hybrid Attention Architectures

*Taxonomy — modern long-context decoders that mix full attention with cheaper "efficient" attention modules.*

**TL;DR:** Pure full attention has $O(n^2)$ cost; pure efficient attention (SWA, Mamba/SSM, linear attention) loses long-range retrieval. The modern compromise is a **hybrid**: a small fraction of full-attention layers carries the global signal, the rest of the stack uses an efficient module to bound per-layer KV and FLOPs. The 2026 *Rethinking* paper argues this design is doing less than people thought: efficient-attention choice mostly controls *how fast* long-context capability emerges; the asymptote is set by the full-attention slices.

**Related taxonomies:** [_moe](_moe.md), [_normalization](_normalization.md)
**Depth files covered here:** [sliding-window-attention](sliding-window-attention.md) · [mla](mla.md) · [multi-head-attention](multi-head-attention.md)

---

## The problem

Long-context decoders are constrained by **two scaling walls** at once:

- **KV cache** grows linearly with sequence length per layer per head. At 100K+ context this dominates serving memory.
- **Attention FLOPs** grow quadratically with sequence length. For prefill, this is the dominant compute cost.

A "long-context model" needs to keep both bounded *without* losing the ability to **retrieve** information from a distant token. Empirically, full attention is the only known module that reliably retrieves at long range; efficient modules are good at modeling but poor at retrieval.

---

## The shared pattern

Every modern hybrid stack mixes two kinds of layers:

1. **A small number of full-attention layers** (1 every $N$, or grouped at top/bottom) that pay the $O(n^2)$ cost but carry long-range retrieval.
2. **A larger number of efficient layers** that bound per-token KV and FLOPs to $O(W)$ or $O(d)$: sliding-window attention, recurrent / SSM mixers, linear attention.

The hybrid is then orthogonal to two further axes:
- **Where the full-attention layers live** (every $N$th layer vs. clustered).
- **What the efficient module is** (SWA vs. Mamba-style SSM vs. linear).

---

## Variants

| Technique | Key idea | Main tradeoff | When it wins |
| --- | --- | --- | --- |
| [sliding-window-attention](sliding-window-attention.md) | local-only attention, fixed window $W$ | poor long-range retrieval alone | paired with full-attention slices (Mistral, Gemma, GPT-OSS) |
| Mamba / SSM mixers (no depth file yet) | recurrent state-space sequence mixer | full-attention layers still needed for needle retrieval | Jamba, Zamba, RecurrentGemma — hybrids with $\le 1/4$ full-attention |
| Linear attention (no depth file yet) | attention with linearized kernel | retrieval lags full-attention; quality lower at scale | early hybrids; mostly displaced by SWA / SSM |
| [mla](mla.md) | full attention with compressed KV cache (low-rank latent) | computation is still $O(n^2)$; memory is the win | when retrieval matters but KV budget is tight (DeepSeek-V3) |
| Sparse / sink attention (no depth file yet) | dense local + a few global anchor tokens | tuning anchor count is tricky | StreamingLLM and friends; long-form generation with bounded cache |

---

## How to choose

- **You can afford the prefill cost but not the KV cache.** Use [MLA](mla.md) — keep $O(n^2)$ attention, shrink the cache 5–10×.
- **You want both prefill *and* KV bounded.** Use SWA + a few full-attention slices. The 2026 *Rethinking* paper notes that the *choice* between SWA and SSM mostly affects training speed, not asymptotic long-context quality — so pick by implementation maturity.
- **You're targeting on-device decode.** Recurrent / SSM mixers (RecurrentGemma, Mamba) give $O(1)$ per-token decode state; pair with at least one full-attention layer to retain retrieval.
- **Don't strip full attention entirely.** Across the hybrid families, retrieval failures track the *absence* of full-attention layers, not the choice of efficient module. The minimum useful ratio appears to be roughly 1 full-attention layer per 4–8 efficient layers.

---

## Adjacent but distinct

- **Pure full-attention models** (LLaMA-3, Qwen2.5 dense) — same operation as hybrid full layers, no efficient module. Simpler; worse KV / FLOPs at scale.
- **MoE** — orthogonal axis: changes the *FFN*, not the attention. A model can be both MoE *and* a hybrid-attention stack (Jamba-MoE, GPT-OSS).
- **Retrieval-augmented models** — externalize the long context entirely into a retriever, leaving the attention stack short. Different design philosophy; pairs sometimes with hybrid bases.

---

## Sources

- Paper: *Rethinking the Role of Efficient Attention in Hybrid Architectures* — Ziqing Qiao et al., Tsinghua / OpenBMB, 2026 — [arXiv:2606.15378](https://arxiv.org/abs/2606.15378). Empirical decomposition of what each layer type contributes.
- Paper: *Mistral 7B* — Jiang et al., 2023 — sliding-window-only decoder (later models added full-attention slices).
- Paper: *Jamba* — AI21 Labs, 2024 — first widely-deployed SSM+attention hybrid.
- Paper: *RecurrentGemma* — Google DeepMind, 2024 — production hybrid recurrent + attention.
- Paper: *DeepSeek-V2 / V3* — DeepSeek, 2024–2025 — MLA-based KV compression as a different point on the same Pareto frontier.

---

## Conventions

- **Filename:** `_hybrid-attention.md` (taxonomy).
- **Folder placement:** `architectures/`.
- **Scope:** designs that mix two or more attention-style modules across depth. Pure-full and pure-MLA are referenced but not the focus; SWA, SSM, and linear attention live here.
