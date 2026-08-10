# GST-Bench
*Depth — VQA benchmark for global spatial awareness in video-understanding VLMs.*

**TL;DR:** GST-Bench (Global-Spatial-Temporal Benchmark) measures whether video VLMs can build a globally consistent scene representation across long viewpoint streams, not just answer questions about the current frame. Ships a paired local benchmark (GST-Bench-Local) that isolates the failure mode: models are strong at local perception but weak at consolidation over time. Human ~79 vs. best zero-shot ~43.

**Prereqs:** [../multimodal/README.md](../multimodal/README.md)
**Related:** [../evaluation/README.md](../evaluation/README.md)

---

## What it is

A VQA benchmark of human-verified questions over 6,790 minutes of synthetically generated video. Questions require (a) accurate spatial inference from novel viewpoints unseen in the input video, and (b) mapping egocentric observations to a global top-down map. The paired GST-Bench-Local uses the same task templates on single or few-viewpoint clips, so the *local minus global* delta cleanly attributes the deficit.

A training corpus (GST-Train) is released alongside for training-side interventions.

## How it works

Task shape:

```
Input:  a long egocentric or third-person video stream + a spatial question
Output: an answer requiring inference across the whole stream
Eval:   exact-match / structured-answer grading vs. human-verified labels
```

Two axes stressed:

- **Novel-viewpoint inference** — the answer requires reasoning about a viewpoint not present in the input frames.
- **Egocentric → top-down mapping** — the answer requires re-projecting observations to a global coordinate frame.

Local counterpart:
- Same task templates, but the model only needs to answer within a single viewpoint. Provides a within-model control for perception ability.

Splits: main eval + local eval + training set.

## Why it matters

Video VLMs are being pitched as embodied-agent backbones (navigation, planning, memory). GST-Bench shows that consolidation over long streams is the missing piece: 22 SOTA VLMs cluster far below human, but the *local* eval is close to human. The gap is not perception, it's memory / integration.

Sets a concrete target for training-time interventions (long-context video pretraining, viewpoint-conditioned objectives) and memory-side interventions (structured scene memory).

## Gotchas & tricks

- Synthetic video simplifies grading but introduces distribution shift from real footage — cross-check on a small real-video subset when using GST scores to greenlight a deployment.
- The local–global gap is meaningful only if the *same* model is scored on both — comparing published local scores against published global scores across models mixes confounders.
- Zero-shot ≠ trained. Fine-tuning on GST-Train materially closes the gap, so headline zero-shot numbers understate what's reachable with modest supervised effort.

## Sources

- Paper: *GST-Bench: Can VLMs Develop Global Spatial Awareness from Video?* — Huang et al., 2026 — [arXiv:2608.05747](https://arxiv.org/abs/2608.05747)
