# Muir (sensitivity-allocated linear attention)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A linear-attention variant in the gated delta-rule (GDN) family. Muir (Google, 2026) exploits an empirical asymmetry — gated delta-rule models are much more sensitive to *write* updates than to *erase* updates — and allocates parameters accordingly: a cheap low-rank shared erase gate, a full-rank write path, and a **matched-filter readout** head that treats retrieval as correlation decoding.

**Prereqs:** [attention](../fundamentals/attention.md), [multi-head-attention](multi-head-attention.md)
**Related:** [mla](mla.md), [transformer-block](transformer-block.md)

---

## What it is

Linear attention compresses the growing KV sequence into a fixed-size state that is updated by a recurrence. The gated delta-rule family (GLA, GDN, GDN-2) parameterises that recurrence as (erase then write):

```
state_t = state_{t-1} * erase_gate(x_t) + write(x_t)
```

In vanilla GDN both gates are full-rank matrices. Muir asks: do both gates need the same capacity?

## How it works

**Asymmetric gate budget.** An ablation in the paper varies the rank of the erase gate vs write gate separately. Perplexity is near-insensitive to erase-gate rank down to very small ranks, but collapses as the write gate is compressed. Muir makes the erase gate a **shared low-rank projection** across heads and keeps the write path full-rank.

**Matched-filter readout.** The output at step t is typically `state_t · q_t`. Muir replaces the straight dot-product with a learned readout gate that is theoretically motivated by the matched-filter view (treat key retrieval as correlation decoding against the compressed state).

**Wider FFN.** Freed parameter budget from the low-rank erase gate is returned to a wider feed-forward.

**Stable chunkwise training.** A reformulation of the recurrence that stays numerically stable under bf16 at long chunk sizes (32K tokens in the paper).

## Why it matters

Linear-attention LLMs (Mamba-2, GLA, GDN) are the credible path to true long-context models — but every variant pays a quality tax vs. full attention. Muir's contribution is parameter *allocation*, not a new recurrence. The asymmetric-sensitivity argument reframes gated recurrences: capacity should follow sensitivity.

Reported results: lower perplexity than GDN-2 and a Transformer baseline across 166M–1.34B parameters, better long-context retrieval, training-speed gains at 32K tokens.

## Gotchas & tricks

- **Shared low-rank erase** shares the projection *across heads*. Per-head low-rank gives up most of the savings.
- **Matched-filter readout** is sensitive to initialisation; the paper uses a specific orthogonal init.
- Chunkwise training numerical stability is non-trivial at long context — follow the paper's rescaling trick exactly.

## Sources

- Paper: Muir: Linear Attention with Sensitivity-Allocated Gating and Matched-Filter Readout — Tabak & Schaefer, 2026 — [arXiv:2611.08376](https://arxiv.org/abs/2611.08376)
