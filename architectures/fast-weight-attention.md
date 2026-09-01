# Fast-Weight Attention (Falcon updates)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Recasts recurrent fast-weight memories and selective state-space models as **online learning rules** under autoregressive prefix-prediction semantics, and derives a normalized-update family — **Falcon-1/2/3** (regression) and their inner-product variants **1A/2A/3A** — with positive-decay renormalization. Numerically stable in recurrent, masked-parallel, and chunk-parallel forms; competitive in language modeling and notably better at length extrapolation on variable-digit addition.

**Prereqs:** [../fundamentals/attention.md](../fundamentals/attention.md), [multi-head-attention.md](multi-head-attention.md)
**Related:** [mla.md](mla.md) · [sliding-window-attention.md](sliding-window-attention.md)

---

## What it is

Linear-attention, fast-weight-memory and selective-SSM variants (Mamba-family and friends) all share a structure: a fixed-size recurrent state is updated per token, and reads produce the output. The paper's contribution is to *derive* the update rule from an explicit **online-learning objective** on the recurrent state — turning the design of these variants into a choice of loss and step-size, not just architectural taste.

## How it works

**Setup.** Under read-after-write autoregressive semantics, the local training example at step $t$ is the **prefix-aligned pair** $(\phi(k_{t-1}), v_t)$. The more common same-step pair $(\phi(k_t), v_t)$ is causal too, but optimizes a different internal objective — a distinction the paper makes explicit.

**Two objective families:**

- **Squared-error regression** → yields the Falcon-1/2/3 update family.
  - **Falcon-1** — scalar NLMS (single normalized step size).
  - **Falcon-2** — per-column extension (per-feature step size).
  - **Falcon-3** — sliding-window mini-batch (small batch of recent pairs).
- **Negative inner-product** → yields Falcon-1A/2A/3A (the "A" siblings).

**Stability.** All variants use **positive-decay renormalization** to keep the recurrent state bounded and the updates numerically well-behaved. Each is written in three equivalent forms — recurrent, masked-parallel, and chunk-parallel — so they can run efficiently on the same kernels the Mamba/linear-attention families already use.

## Why it matters

- Provides a **single scaffold** — temporal alignment, plasticity, forgetting, bounded rehearsal — for reasoning about fast-weight, linear-attention, and SSM variants side-by-side.
- Length extrapolation gain on variable-digit addition is a direct behavioural signal that the update actually generalizes across sequence lengths, not just fits within-training-length.
- Because the derivation is from an online-learning objective, adding e.g. a regularizer, a step-size schedule, or a different loss is a principled move rather than a guess.

## Gotchas & tricks

- The prefix-aligned pair choice matters. Using $(\phi(k_t), v_t)$ (the same-step association) is still causal but optimizes a subtly different criterion — the two look interchangeable and aren't.
- Positive-decay renormalization is doing real work; naïve NLMS without it drifts numerically over long sequences.
- Chunk-parallel form is what unlocks GPU efficiency; the pure recurrent form is only useful pedagogically.

## Sources

- Paper: *Fast Weight Attention for Continual Learning* — Zhang et al., Tsinghua IIIS / Princeton / UCLA, 2026 — [arxiv](https://arxiv.org/abs/2608.27763)
