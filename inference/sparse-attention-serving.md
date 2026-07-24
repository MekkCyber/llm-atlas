# Sparse Attention Serving with Runtime Load Balancing
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Training-free sparse attention (Top-p / Top-k routing) cuts FLOPs but creates uneven per-head workloads under multi-GPU sequence parallelism — the rank with the heaviest heads becomes a straggler and eats the savings. FVAttn combines Top-p + Top-k safety floor + video-aware block organization on the frontend with **runtime load balancing** (P2P migration of heavy heads to shorten the critical path) and **slack-aware augmentation** on the backend. On step-distilled Wan2.2 I2V: load imbalance 1.34 → 1.08, 4.41× attention speedup vs FlashAttention, 2× DiT inference speedup.

**Prereqs:** [../fundamentals/attention.md](../fundamentals/attention.md)
**Related:** [../architectures/multi-head-attention.md](../architectures/multi-head-attention.md), [../systems/dualpipe.md](../systems/dualpipe.md)

---

## What it is

Video diffusion transformers (DiTs) process long spatio-temporal token sequences — 100k+ tokens is common — and spend most of inference time in self-attention. Training-free sparse attention (each query attends to only a fraction of keys, chosen adaptively per head) is the standard cost-cut, but it interacts badly with **sequence parallelism**: each GPU holds a slice of the sequence, and the mask sparsity varies per head. Some ranks get a dense-ish workload, others get a sparse one. The step time is the max-over-ranks — so the FLOP savings get canceled by rank-level stragglers.

FVAttn treats the imbalance as a **scheduling problem** on top of the sparse-routing frontend.

## How it works

**Frontend (sparse-routing).**
- Top-p routing per head (attend to enough keys to cover probability p).
- Top-k **safety floor** so no head ever routes to fewer than k keys (avoids catastrophic misses when Top-p under-samples).
- Video-aware block organization — group tokens by spatial neighborhood so per-block routing decisions are coherent.

**Backend (distributed execution).**
- **Runtime Load Balancing.** After routing, per-head compute cost is measured. A small number of the heaviest heads are migrated via P2P communication to under-loaded ranks. This directly shortens the critical path (max-over-ranks time).
- **Slack-Aware Sparse Augmentation.** Ranks below the critical path have slack — FVAttn fills that slack with *additional* high-value attention blocks (blocks that were pruned by Top-p but had non-trivial score). Quality goes up, wall-clock stays the same.
- **Overlap** hides P2P migration and scheduling behind existing compute.

## Why it matters

Sparse attention has been the theoretical answer for video-DiT serving cost for years, but the practical gains stall on distributed stragglers. FVAttn is the first serving system to attack the imbalance directly, and the numbers are large enough (4.41× attention speedup, 2× end-to-end DiT) that this changes the deployment economics for video-generation services.

Directly usable inside vLLM/SGLang-style serving stacks once they add sequence parallelism for DiTs.

## Gotchas & tricks

- The Top-k safety floor is not optional. Pure Top-p can produce degenerate sparsity on some heads (very peaked distributions) and lose quality.
- P2P migration is only worth it when the migration cost is smaller than the critical-path savings — the paper triggers it only for the top few heaviest heads.
- Sparse-attention quality regressions are subtle in video — check temporal consistency and identity preservation, not just per-frame quality.
- Evaluated on step-distilled diffusion (Wan2.2 I2V). Vanilla many-step diffusion has different bottlenecks; results should be re-measured.

## Sources

- Paper: *FVAttn: Adaptive Sparse Attention with Runtime Load Balancing for Video Generation* — Liu et al., 2026 — [arXiv:2607.16190](https://arxiv.org/abs/2607.16190)
- Related: FlashAttention (baseline in the paper's speedups).
