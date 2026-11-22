# Mixture of Block Attention (LightningMoBA)

*Depth — sparse-attention variant that routes each query to a dynamically chosen subset of KV blocks, with a kernel schedule that overlaps selection and compute.*

**TL;DR:** Mixture of Block Attention (MoBA) scores KV blocks against each query and runs softmax attention only over the top-scoring ones. LightningMoBA keeps MoBA's scoring rule and softmax semantics but redesigns the inference stack: **threshold-based dynamic routing** (per-query count instead of fixed Top-k), **KV-centric work ownership** with fanout-aware splitting, and **Hopper pipelining**. End-to-end 1.20×–1.60× generation speedup over FlashMoBA at matched Top-16 retention, with mean RULER rising from 74.42 → 75.28 — speedup *and* a small quality gain at long context.

**Prereqs:** [../fundamentals/attention.md](../fundamentals/attention.md), [../architectures/_moe.md](../architectures/_moe.md)
**Related:** [README.md](README.md)

---

## What it is

The KV-length analogue of MoE: instead of routing each token to a subset of FFN experts, route each query to a subset of KV blocks. The scoring rule picks blocks by a cheap affinity metric over a block summary; softmax attention runs only on the retained blocks. LightningMoBA is the execution-level rebuild of this idea that unlocks its speedup potential on H100.

## How it works

Three ingredients:

**Threshold-based dynamic routing.** Classical MoBA keeps the top-k blocks per query with k fixed. LightningMoBA instead keeps blocks whose score exceeds a threshold — so the retained-block count *varies per query*. Easy queries need a few; hard ones get more. Avg retention is matched to Top-k for comparison, but variance shifts compute toward queries that need it.

**KV-centric execution with fanout-aware splitting.** Work ownership is organized around **KV blocks**, not queries. A query scatters into every KV block it selected; a KV block gathers all queries that picked it. For long sequences, hot blocks are split and processed in parallel without reloading their KV tiles. Memory bandwidth on reused KV goes down sharply.

**Hopper pipelining.** The selection pass and the attention compute are overlapped using TMA / async-copy on H100. The selection pass produces per-block routing lists while the previous block's attention is still computing.

The scoring rule and the softmax normalization are unchanged — LightningMoBA is a **kernel and scheduling** story, not a new attention variant.

## Why it matters

Sparse-attention variants live or die by their kernel implementation. MoBA was a reasonable scoring rule whose generation-time wins were masked by execution overhead. LightningMoBA shows the headroom was in execution strategy — a common story for sparse attention at generation time. Also: the small RULER gain alongside the speedup is unusual for sparse attention, which usually trades a point of quality for speed.

## Gotchas & tricks

- Threshold routing requires a per-model threshold sweep; too low and avg retention balloons, too high and quality drops.
- KV-centric execution assumes long enough sequences to amortize the gather — short-context workloads see a smaller win.
- Hopper-specific features (TMA, async copy) do the heavy lifting; porting to Ampere loses a chunk of the speedup.
- The scoring rule's cost itself is non-trivial at very long context; the pipelining is what hides it.

## Sources

- Paper: *Accelerating Mixture of Block Attention via Dynamic Routing, KV-Centric Execution and Runtime Pipelining* — Tencent LightningMoBA team, 2026 — [arXiv:2611.26268](https://arxiv.org/abs/2611.26268). Code: [github.com/Tencent/LightningMoBA](https://github.com/Tencent/LightningMoBA).
- Related: MoBA (Mixture of Block Attention) — the scoring-rule origin; FlashMoBA — baseline implementation.
