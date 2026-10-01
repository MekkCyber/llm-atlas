# KV-Cache Compression
*Depth — how long-context serving systems shrink the KV cache, and the retrieval failure modes it creates.*

**TL;DR:** Long-context inference is bottlenecked on KV-cache memory and attention cost. Serving systems compress the cache by chunking consecutive tokens into fewer entries, keeping sliding windows + attention sinks, or learning compressed summaries. **Zhu et al.'s 2026 "Periodic Weak Spots"** paper shows chunked compression introduces **phase-dependent retrieval**: the same fact can be easy to recover at one source phase and hard at another, with gaps up to **40 pp** — and traces this to attention-head *phase specialization*. Long-context benchmarks that don't vary phase overestimate serving systems.

**Prereqs:** [README.md](./README.md), [../fundamentals/dca.md](../fundamentals/dca.md)
**Related:** [../interpretability/README.md](../interpretability/README.md), [../evaluation/README.md](../evaluation/README.md)

---

## What it is

The KV cache stores one (key, value) tensor pair per layer per token. At 128K context, this is tens of GB per request, and attention cost grows linearly in context size. Compression schemes trade exact recall for throughput by storing *fewer* KV entries:

- **Chunked compression** — merge $k$ consecutive tokens into one entry (fixed stride).
- **Sliding window + sink** — keep the first few and most recent tokens exactly, drop the middle.
- **Hybrid** — chunked middle + exact sink + exact recent window.
- **Learned compression** — a small model predicts compressed KV summaries.

All these schemes assume retrieval is roughly equivariant across the sequence. Periodic Weak Spots shows that assumption is false under chunked compression.

## How it works

### Chunked compression + phase

Define the **source phase** of a token as its position within a chunk: a token at offset $p \in \{0, 1, \ldots, k-1\}$ inside its chunk has phase $p$. Phases are periodic along the sequence.

The 2026 paper pretrains transformers with controlled chunked compression and measures retrieval accuracy conditioned on the source phase. The gap between best and worst phase on the same compression scheme reaches **40 pp**. The pattern persists across model sizes and chunk strides.

### Phase specialization in attention heads

Mechanistic analysis shows attention components (heads and MLP mixing patterns) specialize to specific phases: head $h$ reliably attends to phase-$p$ sources, head $h'$ to phase-$p'$ sources. When a phase is under-covered by any head, retrieval at that phase collapses.

Implication: chunked compression doesn't degrade retrieval uniformly — it degrades it *periodically*, and the period is tied to the compression stride.

## Why it matters

- **Long-context benchmarks under-measure.** Needle-in-haystack probes that place needles at the same offset mod chunk-stride systematically miss weak-spot phases. Robust evals must vary needle phase or report worst-phase accuracy.
- **Hybrid schemes aren't a free lunch.** Chunked + sink + sliding-window still has chunked phases in the compressed middle; the sink and sliding window don't rescue retrievals from there.
- **Argues for pretraining-aware compression.** If compression is applied only at serving time, phase specialization has not been learned around it. Pretraining under compression (or distilling into a compressed-KV model) may be the only way to avoid worst-phase cliffs.

## Gotchas & tricks

- **Needle placement matters.** Re-run your long-context eval with the needle at every phase in `{0, …, stride−1}`. The spread is the real signal, not the average.
- **Stride vs. window interactions.** Smaller strides reduce the per-phase gap but increase cache size. The usual tradeoff curves need to be re-measured per phase.
- **Head pruning amplifies the effect.** Dropping heads for inference efficiency can delete the exact heads that owned a phase. Measure phase accuracy after pruning.
- **The result is for *pretrained* chunked compression.** Learned compression and summary-based schemes may distribute phase differently; the paper does not generalize to them.

## Sources

- Paper: *Periodic Weak Spots: Phase Sensitivity from Chunked KV-Cache Compression* — Zhu, Yi, Cheng, Lv, Liu, Ying, Ma, Dong — Tsinghua / ByteDance, 2026 — [arXiv:2609.36322](https://arxiv.org/abs/2609.36322).
