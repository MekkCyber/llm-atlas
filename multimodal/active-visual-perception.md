# Active Visual Perception (Zoom-in Agent)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** For long, visually rich documents, treat **visual resolution as an adaptive reasoning-time resource** instead of encoding every page at full resolution. The VLM starts from a low-resolution view of all pages and, guided by the current question and its intermediate reasoning, **selectively zooms into high-resolution crops** of the regions that matter. Trained with SFT (17.9K zoom trajectories) + RL (19.2K hard examples) so that region-choice is learned end-to-end. Introduced as **InSight-doc** (HKUST / Huawei, 2026).

**Prereqs:** [README.md](README.md)
**Related:** [../post-training/rlvr.md](../post-training/rlvr.md), [../agents/README.md](../agents/README.md)

---

## What it is

Two dominant approaches to long-document VLMs today:

1. **Full-resolution everything.** Encode every page at high resolution and stuff the tokens into context. Expensive, context-rot-prone, latency-heavy.
2. **External retriever.** Bolt a document retriever on top of a smaller-context VLM. Adds a subsystem and a training/inference stack that must be kept aligned.

Active perception is a middle path: **the model itself decides which regions to re-encode at higher resolution**, based on the question and its intermediate reasoning. Zoom is a tool call — a structured action the model emits, executed by the pipeline, whose output (higher-res crop tokens) is appended to context for the next reasoning step.

## How it works

1. **Low-res global view.** All pages are encoded at low resolution and placed in context.
2. **Emit a zoom action.** The model outputs a structured action like `zoom(page=k, bbox=(x0,y0,x1,y1))` when it needs finer evidence.
3. **Executor returns a high-res crop.** The pipeline crops the requested region at high resolution, encodes it, and appends the resulting tokens to context.
4. **Iterate until answer.** The model reasons over the accumulating evidence and eventually emits an answer.

**Training corpus:**
- **SFT:** 17.9K high-quality trajectories with region-level zoom-in actions.
- **RL:** 19.2K hard examples, using verifiable-reward RL on the final VQA answer.

Both stages teach the model *when* to zoom and *where*, without an external retriever.

## Why it matters

- **Big accuracy lift.** InSight-doc-8B improves the baseline by **+4.3–16.4 accuracy points** on document VQA benchmarks.
- **Hallucination and latency both drop.** On long documents, **>40% reduction in hallucination** and **41–68% reduction in inference latency**, while maintaining an accuracy lead over full-resolution encoding.
- **No retriever subsystem.** The zoom decision is internal to the VLM — one training stack, one inference pipeline.
- **Generalizes beyond documents.** The pattern — "vision is a tool the model calls" — extends to charts, code screenshots, medical imagery, any high-resolution visual data where full-res-everywhere is unaffordable.

## Gotchas & tricks

- **Zoom action schema matters.** Coarse actions (page-only) cap the resolution gain; overly fine actions bloat the action space. The paper's `page + bbox` granularity is a reasonable default.
- **Trajectory data is the bottleneck.** Building the 17.9K SFT trajectories with sensible zooms is the expensive step; scripts that generate synthetic zoom paths from OCR + question analysis help.
- **RL reward has to be verifiable.** Free-form QA rewards need a judge; the paper stays with tasks where the answer can be checked exactly.
- **Context still grows with zooms.** Aggressive zooming can defeat the latency gain; a soft cap on zoom count per query is prudent.

## Sources

- Paper: *InSight-doc: Agentic Visual Perception for Long-Document Understanding* — Li, Xie, Yao, Wu, Hong, Huang, Zhang (HKUST / Huawei), arXiv 2608.10628, 2026.
- Code: https://github.com/m-Just/InSight-doc
