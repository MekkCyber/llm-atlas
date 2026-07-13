# Delta Rule (for Linear Attention)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A **recurrent update rule** for the state matrix of a linear-attention layer, borrowed from classical associative memory (Widrow–Hoff / Krotov–Hopfield). Given a key–value pair $(k_t, v_t)$, the state $S$ is updated by *subtracting the current readout for $k_t$ and adding the new value* — a delta correction. This gives modern linear-attention variants (DeltaNet, Gated DeltaNet, Kimi Delta Attention, Gated DeltaNet-2) meaningfully better recall than the original Katharopoulos linear attention while preserving $O(L)$ cost.

**Prereqs:** [attention.md](./../fundamentals/attention.md), [multi-head-attention.md](./multi-head-attention.md)
**Related:** [_linear-attention.md](./_linear-attention.md) · [mla.md](./mla.md)

---

## What it is

A structured way to write into the recurrent state of a linear-attention layer. In plain linear attention the state accumulates outer products $S_t = S_{t-1} + \phi(k_t) v_t^\top$ — every write is additive, so there is no mechanism to *overwrite* an old association when a new one arrives with the same key. That's the recall gap that made plain linear attention weak.

The delta rule fixes this by turning each write into a **correction**: read what the state currently associates with $k_t$, compute the delta between that and the new $v_t$, and write the delta. Old associations are supplanted rather than piled on top of.

## How it works

Let $S_{t-1} \in \mathbb{R}^{d_k \times d_v}$ be the state before token $t$, with $k_t, v_t \in \mathbb{R}^{d_k}, \mathbb{R}^{d_v}$.

**Read** what the state currently associates with $k_t$:

$$
\hat v_t = S_{t-1}^\top k_t
$$

**Write** the correction (the "delta"):

$$
S_t = S_{t-1} + k_t (v_t - \hat v_t)^\top
$$

**Readout** for the query is unchanged:

$$
o_t = S_t^\top q_t
$$

This is the Widrow–Hoff delta rule applied online, one token at a time. It preserves $O(L)$ compute (constant work per step) and $O(1)$ state size.

### Gated variants

Modern implementations add per-token **gates** that decide how much to write:

$$
S_t = (1 - g_t) \odot S_{t-1} + g_t \odot \bigl[ S_{t-1} + k_t (v_t - \hat v_t)^\top \bigr]
$$

The gate $g_t$ is a small learned function of the token (often a sigmoid over $q_t$/$k_t$). This is what turns DeltaNet into Gated DeltaNet, Kimi Delta Attention, and Gated DeltaNet-2 — they differ in *where* the gates sit (input side, output side, both) and *how* they're computed.

### Parallel training

The recurrence looks strictly serial, but the paper *Parallelizing Linear Transformers with the Delta Rule Over Sequence Length* (Yang et al., 2024) shows it can be rewritten as a chunk-parallel algorithm on GPU: split the sequence into chunks, compute intra-chunk updates via block-triangular matmul, and stitch chunks with a scan. This is what makes delta-rule variants trainable at LLM scale.

## Why it matters

- **Closes the recall gap** of plain linear attention while keeping $O(L)$ cost.
- **Unifies a whole family.** DeltaNet, Gated DeltaNet, Kimi Delta Attention, and Gated DeltaNet-2 are variations on gate placement over this same recurrence.
- **Composes with hybrids.** In cross-layer routing schemes (a handful of softmax layers, the rest delta-rule), the delta rule is the workhorse recurrence.
- **Different lever than sparse/MLA.** Not competing with cheaper softmax — orthogonal. You could imagine a stack that mixes MLA layers with delta-rule layers.

## Gotchas & tricks

- **Feature map still matters.** Whether keys go through an ELU, RMS-norm, or identity before the delta update meaningfully changes stability. Different papers pick differently — check the specific implementation.
- **Gate placement is load-bearing.** Input-side gates control what gets written; output-side gates control what gets read. Papers disagree; the ETH 2026 study finds routing matters more than gate placement.
- **State grows in norm.** Long sequences accumulate corrections; RMS-normalizing the state or the readout is common to keep numerics stable.
- **Recall on synthetic tasks vs. natural language.** Delta-rule variants beat plain linear on associative-recall probes but still trail softmax on hard natural-language recall — one reason hybrid stacks exist.

## Sources

- Paper: *Parallelizing Linear Transformers with the Delta Rule Over Sequence Length* — Yang et al., 2024 — the DeltaNet paper.
- Paper: *Gated Delta Networks* — Yang et al., 2024 — adds gates.
- Paper: *Linear Attention Architectures: Mechanisms, Trade-offs, and Cross-Layer Routing* — Cerruti et al., 2026, [arXiv 2607.07953](https://arxiv.org/abs/2607.07953) — comparative study covering DeltaNet, GDN, KDA, GDN-2 under one delta-rule frame.
- Related paper: *Dense Associative Memory Is Robust to Adversarial Inputs* — Krotov & Hopfield, 2016 — the associative-memory motivation.
- Classical: Widrow & Hoff, 1960 — the original delta rule from adaptive filtering.
