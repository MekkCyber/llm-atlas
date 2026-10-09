# Sliding-Window Linear Attention (SWLA)

*Depth — a hybrid attention variant that borrows SWA's positional inductive bias into gated linear attention, closing the length-extrapolation gap between SWA and LA hybrids.*

**TL;DR:** Hybrid LLMs that interleave full attention with sliding-window attention (SWA) and those that interleave full attention with gated linear attention (LA) exhibit a "Seesaw Effect": **LA hybrids benefit more from long-context continual pretraining; SWA hybrids extrapolate better to lengths beyond training.** The paper attributes the gap to positional inductive bias — SWA inherits RoPE's monotonic distance decay, LA does not. SWLA explicitly imports a sliding-window prior into LA. Reaches 16× training-free length extrapolation with 100% NIAH-SK1 accuracy at 64k context.

**Prereqs:** [multi-head-attention](multi-head-attention.md), [rope](../fundamentals/rope.md), [_positional-encoding](../fundamentals/_positional-encoding.md)
**Related:** [mla](mla.md) · [transformer-block](transformer-block.md)

---

## What it is

A linear-attention variant whose kernel is masked by a sliding window of radius $w$:

$$
\mathrm{SWLA}(Q, K, V)_t = \sum_{t - w \le s \le t} \phi(Q_t)^\top \phi(K_s) \cdot V_s
$$

Standard gated linear attention sums over *all* previous positions with a learned gate; SWLA caps the receptive field at $w$ (typical $w = 256 – 1024$). The result is that each token attends only to a bounded local window, inheriting the positional bias that pure SWA has.

In a hybrid stack, SWLA layers interleave with full attention layers — same macro-structure as SWA- or LA-hybrids, with SWLA replacing one of them.

## How it works

### The Seesaw Effect

The paper measures long-context behaviour on two axes:

- **In-length performance** (train on 32k, evaluate at 32k) — LA hybrids win after continual long-context pretraining.
- **Out-of-length performance** (train on 32k, evaluate at 64k, 128k) — SWA hybrids win, often by large margins.

The authors argue the cause is positional inductive bias: SWA inherits RoPE's monotonic distance decay (close tokens attend more), which transfers cleanly to unseen lengths. LA has no such bias and must learn position from data.

### The fix

SWLA imports the window bias into LA. Mechanistically it's a cheap kernel mask on top of the existing LA formulation; no new parameters.

### Measured outcomes

- **16× training-free length extrapolation**: train at 4k, evaluate at 64k.
- **100% NIAH-SK1 accuracy at 64k**, matching the best SWA hybrids.
- Preserves LA's long-context *continual pretraining* benefit (the thing SWA hybrids were worse at).

## Why it matters

Hybrid LLMs (Jamba, Falcon-Mamba, Zamba, MiniMax-01) are now mainstream production architectures. The LA vs. SWA choice has been empirical; SWLA offers a principled answer and a hybrid that beats both on their respective strengths.

Also the first mechanistic explanation of why pure-linear-attention models struggle to extrapolate without extensive long-context training — "no positional inductive bias" rather than "RNN memory decay".

## Gotchas & tricks

- **Window size $w$ matters.** Too small and the layer loses access to global structure; too large and the bias dilutes. The paper sweeps between 256 and 1024.
- **Not a replacement for the full-attention layer.** Hybrid stacks still need at least one full-attention layer per block for true long-range dependencies.
- **Positional-encoding alignment.** The window mask does the job of RoPE's decay; using RoPE on top may be redundant or counterproductive.
- **Training-free extrapolation is specifically for the SWLA layer** — the full-attention layers' extrapolation depends on their own positional scheme.

## Sources

- *Mechanics of Long-Context Hybrid Models: Part 1.1 — From Hybrid Attention to Hybrid Position* — Liu, He, Qiu — Fudan University — 2026 — [arXiv:2610.10114](https://arxiv.org/abs/2610.10114)
