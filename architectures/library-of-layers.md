# Library of Layers (LoL)
*Depth — a transformer that stores a shared library of blocks and routes each input through a per-token layer ordering.*

**TL;DR:** LoL keeps `K` candidate transformer blocks in a shared library and uses a small router to pick, per token and per depth position, which block to apply. The library size `K` is a single scalar that interpolates smoothly between parameter reuse (small `K`, Universal-Transformer-style weight sharing) and sparse compute (large `K`, MoE-style sparsity). The routing paths turn out to be legible — they correlate with input language, with reasoning-puzzle class, and with safety labels — and forcing a path can steer model behavior.

**Prereqs:** [transformer-block.md](./transformer-block.md) · [_moe.md](./_moe.md)
**Related:** [multi-head-attention.md](./multi-head-attention.md) · [aux-loss-free-balancing.md](./aux-loss-free-balancing.md)

---

## What it is

A standard transformer is a fixed stack `L_1 ∘ L_2 ∘ … ∘ L_N` with one block per depth position. LoL replaces that with a *library* of `K` blocks and, for each depth position `d ∈ {1, …, N}` and token `t`, a router pick `r(t, d) ∈ {1, …, K}`. The chosen block is applied. All depth positions draw from the same library — weights are *shared across depth*, and specialization comes from how the router composes them.

The library size `K` sweeps between extremes:
- `K = 1`: one block repeated `N` times. A looped transformer / Universal Transformer.
- `K = N`: distinct block per position — a standard transformer.
- `1 < K < N`: partial sharing. A small `K` with longer stack `N` is the interesting regime.

LoL is *not* an MoE: MoE routes over width (parallel experts within a layer); LoL routes over depth (sequence of layers from a library).

## How it works

- **Router.** A lightweight linear head over the current token's hidden state outputs scores over the `K` library blocks. Top-1 (or small top-K) selection. The router sees position identity `d` as an explicit input, so the same token can pick different blocks at different depths.
- **Load balancing.** Needs the same balancing pressure as MoE routers — an auxiliary loss, or aux-loss-free bias, to prevent library collapse (one block absorbing most calls).
- **Training.** Standard autoregressive loss. Router is trained end-to-end through a straight-through or Gumbel estimator.
- **Inference.** Each token's forward pass applies `N` blocks drawn from a library of size `K`. Compute is identical to a dense transformer of depth `N` with per-block parameter count equal to the library's block size.

## Why it matters

- **Unified architecture.** Looped transformers and depth-routed MoE collapse into one family parameterized by `K`.
- **Interpretable paths.** Routing sequences correlate with input language (different paths for English vs Chinese), puzzle class (sudoku vs. maze), and safety labels. Paths can serve as a cheap control/interpretation surface — forcing a path steers the output.
- **Parameter efficiency.** Many fewer unique parameters than a dense stack of the same depth for small `K`; retains most of its quality because the router composes a path tailored to each input.

## Gotchas & tricks

- **Router collapse is the first failure mode.** Without balancing pressure, one block dominates and the model degenerates to a Universal Transformer.
- **Depth-position conditioning matters.** Without `d` as an explicit router input, the router has no way to learn "early layers do X, late layers do Y" and quality drops.
- **Not a free lunch on throughput.** Unlike MoE, LoL doesn't reduce compute per token at inference — it reduces parameter count, not flops.
- **Path = control surface.** Because forcing a path changes behavior, LoL offers an interpretation handle at no extra cost — but also a new attack surface for jailbreaks (prompt-level path flips).

## Sources

- Paper: *Library of Layers (LoL): Adaptively Ordering Transformer Layers at Runtime* — McLeish et al., EPFL / NYU / Columbia / UMD, 2026 — [arXiv:2611.03526](https://arxiv.org/abs/2611.03526).
- Paper: *Universal Transformers* — Dehghani et al., 2018 — the `K = 1` extreme.
- Paper: *Mixture-of-Experts with Expert Choice Routing* — Zhou et al., 2022 — closest MoE-side cousin.
