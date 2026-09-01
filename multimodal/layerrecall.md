# LayerRecall
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A **layer-selective memory router** for autoregressive video diffusion. Analyzing a video DiT reveals layers differ sharply in preference for current, recent, and distant context. LayerRecall retrieves historical K/V states and injects them **only into the small subset of "memory-sensitive" layers**, leaving local attention untouched elsewhere. Trained with **Cross-Horizon Prediction Matching (CHPM)**, which uses a privileged long-context reference to supervise the bounded-memory router in prediction space — sidestepping the need for scarce long-video labels.

**Prereqs:** [README.md](README.md), [../fundamentals/attention.md](../fundamentals/attention.md)
**Related:** [../architectures/README.md](../architectures/README.md)

---

## What it is

Long autoregressive video diffusion generates chunk by chunk from a bounded recent context. Pure recency caching preserves local continuity but evicts historical cues — when a subject reappears three shots later, its appearance drifts. Prior memory mechanisms *expose* long-range history to every layer, but the paper's analysis shows most layers don't want it; they perform best on local context. Blanket injection wastes compute and often hurts.

## How it works

**Layer preference analysis.** A per-layer diagnostic categorizes each DiT layer as **current-preferring**, **recent-preferring**, or **distant-preferring**. Only distant-preferring layers benefit from a memory injection.

**Current-conditioned, layer-selective router.** For each new chunk:

1. From the current chunk's queries, retrieve relevant historical K/V states.
2. **Route** those K/V *only* into the identified memory-sensitive layers. Every other layer runs standard local attention.
3. Inference overhead is negligible relative to the base model since only a fraction of layers see memory.

**Cross-Horizon Prediction Matching (CHPM).** To train the router without long-video labels, run a **privileged long-context reference** alongside the bounded-memory student and supervise the student's predictions to match the reference's, in prediction space rather than through direct memory-allocation labels.

## Why it matters

- Best overall on **MemoBench** and **MovieBench** across 100 multi-shot prompts, while matching backbone quality on **VBench-Long** — long-range gain without local regression.
- Enables **memory-guided self-correction**: initially mismatched local attributes return to their historical appearance without resetting ongoing motion or scene structure.
- Cross-backbone portable and adds negligible inference cost — the layer-selective routing is the reason.

## Gotchas & tricks

- Which layers are "memory-sensitive" is backbone-dependent; the diagnostic pass has to be rerun when switching architectures.
- The privileged reference (CHPM teacher) can itself be modest — it just needs the full history, not the best possible model.
- Blanket K/V injection into all layers is a common baseline; skip it — the analysis in the paper explains why it hurts.

## Sources

- Paper: *LayerRecall: A State-Conditioned Memory Router for Long-Horizon Consistency in Video Generation* — Ding et al., Zhejiang University, 2026 — [arxiv](https://arxiv.org/abs/2608.28460)
