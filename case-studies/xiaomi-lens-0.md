# Case Study: Xiaomi-Lens-0

*A 4B multimodal LLM that folds pixel-level segmentation into the autoregressive stream — no separate segmenter, no bolted-on head. Mask-token hidden states drive a shared decoder that reads vision-encoder patches alongside the LLM's own thinking state, so a single mask loss updates the whole backbone.*

**Related concepts:** [mask-token-segmentation](../multimodal/mask-token-segmentation.md) · [qwen2-5](qwen2-5.md)

---

## What this is

**Xiaomi-Lens-0**, released October 2026 by Xiaomi. A 4B-parameter MLLM built on **Qwen3.5-4B** that extends the model with pixel-level perception: it predicts bounding boxes **and segmentation masks** as part of its normal autoregressive output, interleaved with reasoning.

The architectural novelty is in how masks are produced. Instead of attaching a segmentation decoder to a frozen vision encoder, Xiaomi-Lens-0 introduces a special `<|mask|>` token in the response stream. Each occurrence of the token carries a hidden state produced by the LLM; a shared mask decoder reads that state alongside the vision encoder's patch features and emits a mask. The mask loss flows back through the LLM, so pixel-level supervision reaches the representations used for language generation. This is the *shared-representation mask supervision* idea (see [mask-token-segmentation](../multimodal/mask-token-segmentation.md)).

The paper also introduces **LensBench**, a referring/segmentation benchmark that stresses explicit vs implicit wording and multi-target cases across general, aerial, driving, and panoramic imagery.

---

## Architecture at a glance

```
Backbone:        Qwen3.5-4B (text + vision, decoder-only)
Vision encoder:  shared with Qwen3.5 — patch features reused for segmentation
Response stream: text, boxes (<bbox>[x1,y1,x2,y2]</bbox>), and masks (<mask><|mask|></mask>)
                 reasoning wrapped in <think>…</think>
Mask decoder:    lightweight; fuses
                  (1) patch-level features from the vision encoder
                  (2) the LLM's hidden state at each <|mask|> token
                 one shared decoder for all mask tokens
Generation:      standard autoregressive, up to 16,384 tokens at eval
```

### What's shared, what's new

- **Shared with Qwen3.5:** the LLM backbone, the vision encoder, the tokenizer, the vision-to-LLM projection.
- **New:** the `<|mask|>` token, the mask decoder that consumes `<|mask|>` hidden states, the training recipe that enables REC/RES/VQA in one stream, and the data pipeline that assembles paired box-mask-expression supervision.

### Thinking with visual latents

A chain of thought can emit multiple `<|mask|>` tokens — one per region of interest — and each produces a mask. The hidden state carried by the mask token is the "visual latent" the decoder reads; because the latent is produced by the LLM in the middle of reasoning, mask prediction is **conditioned on the ongoing chain of thought**. Mask supervision then updates the backbone through those latents, aligning its mid-reasoning representations with pixel-level signal.

---

## Training recipe

Full stage recipe is partial in the paper excerpt; what is in scope:

### Stage 1 (not detailed in the excerpt available)

Not described in the content retrieved for this case study.

### Stage 2 — grounded REC / RES / VQA

Three streams of synthesized training data, filtered aggressively:

- **Referring expressions (REC/RES).** Object discovery produces box–description pairs; a crop–text verifier filters mismatches. Masks come from a mask generator. Instructions are synthesized from both masked views (target only, background black) and full images, with several expressions per target. Final retention requires a *re-grounding* check: the trained path must return exactly one box with IoU ≥ τ against the candidate, and a semantic check must confirm uniqueness.
- **Grounded VQA.** An MLLM generates counting / attribute / spatial-relation questions. Rule-based filters prune malformed or label-leaking questions; options are shuffled; an independent checker must reproduce the reference answer.
- **Grounded reasoning.** A reasoning model produces `<think>` traces that reference regions via `<bbox>`. The shortest *passing* trace is kept. Each bbox is paired with an existing instance mask or a SAM-3-generated one, accepted when the mask's bounding box agrees with the referenced box above an IoU threshold.

Named datasets: COCO 2017, RefCOCO, RefCOCO+, RefCOCOg, RoboRefIt, ReasonSeg.

### Final checkpoint

The reported model averages **three Stage 2 checkpoints** with equal weights (model souping, same spirit as [model-souping](../pre-training/model-souping.md)).

### What the paper excerpt does not disclose

Loss weighting between mask / box / text terms, optimizer settings, learning-rate schedule, batch sizes, and stage lengths are not available from the parts of the paper fetched. The core novel supervision mechanism is clear; the training-recipe details beyond "three Stage 2 checkpoints averaged" are not.

---

## Key results

### Referring expression comprehension (Prec@0.5)

| Split | Xiaomi-Lens-0 | Qwen3.5-4B (backbone) |
| --- | --- | --- |
| RefCOCO val / TestA / TestB | 94.6 / 96.2 / 92.3 | 90.4 / 92.3 / 87.1 |
| RefCOCO+ val / TestA / TestB | 92.1 / 94.9 / 88.5 | — |
| RefCOCOg val / test | 92.5 / 92.4 | — |

**Average +5.6 points over backbone** across the eight REC splits. On the harder F1@mIoU metric, gains over backbone are **+8.8 / +11.6 / +9.5** on the three families.

### Referring expression segmentation (cIoU)

| Split | Xiaomi-Lens-0 | Qwen3-VL-Seg-4B |
| --- | --- | --- |
| RefCOCO val / TestA / TestB | 83.4 / 84.6 / 81.2 | — |
| RefCOCO+ val / TestA / TestB | 79.9 / 83.3 / 76.3 | — |
| RefCOCOg val / test | 81.1 / 81.2 | — |
| **Average** | **81.4** | 78.6 |

### Multi-instance grounding (COCO 2017, F1 @ mIoU)

**63.9**, vs 63.0 for GroundingPI and 60.6 for GroundingDINO.

### Reasoning + embodied (gIoU / cIoU)

- **ReasonSeg** test 68.0 / 62.5, val 71.2 / 68.5.
- **RoboRefIt** TestA 87.0 / 88.4, TestB 81.5 / 81.2.

### General perception (Xiaomi-Lens-0 vs Qwen3.5-4B)

| Benchmark | Xiaomi-Lens-0 | Qwen3.5-4B |
| --- | --- | --- |
| MME-RealWorld-en | 69.7 | 66.5 |
| MME-RealWorld-cn | 67.5 | 60.3 |
| CV-Bench | 88.1 | 85.4 |
| V\* | 87.4 | 82.2 |
| HR-Bench 4K / 8K | 82.5 / 80.6 | 81.5 / 75.6 |
| CountQA | 50.6 | 43.8 |
| ColorBench | 84.4 | 82.7 |

Perception improves across the board — no regression from the added segmentation head, which the paper's ablation attributes to shared supervision rather than specialist bolt-ons.

### LensBench (new benchmark)

Mask cIoU / box Prec@0.5 by slice:

- Explicit instructions: 71.4 / 85.0. Implicit: 54.7 / 70.0.
- General imagery: 63.9 / 79.1. Aerial: 57.1 / 79.6. Driving: 68.6 / 79.4. Panoramic: 62.3 / 72.1.
- Xiaomi-Lens-0 leads cIoU on **aerial, driving, panoramic**. Qwen3.8-Max leads on general imagery (74.2 cIoU).
- Hardest slice: panoramic + multi-target + implicit (Prec@0.5 48.7, cIoU 48.9).

---

## Ablations

### Spatial supervision (Table 8 in the paper)

| Variant | RefCOCO | RefCOCO+ | RefCOCOg | MME-RW-en | MME-RW-cn | CV-B |
| --- | --- | --- | --- | --- | --- | --- |
| Backbone | 90.4 | 84.8 | 88.0 | 66.5 | 60.3 | 85.4 |
| BBox-only | 93.5 | 89.7 | 90.2 | 68.8 | 66.1 | 87.9 |
| BBox + Mask | 94.6 | 92.1 | 92.5 | 69.7 | 67.5 | 88.1 |

Box supervision alone captures most of the REC gain. Mask supervision adds **+1.1 to +2.4 REC points** and **+0.2 to +1.4 perception points** on top. The headline is that pixel-level supervision does not come at a cost to language-side perception.

### Grounded reasoning (Table 9)

Grounding-only vs grounding + reasoning traces: adding reasoning gives **+0.1 to +0.6 REC** and **+0.5 to +0.8 RES cIoU**. Modest but consistent across all three families.

---

## Why this paper matters

- **Design pattern for open VLMs.** Pixel-level outputs can be absorbed through token-level supervision instead of specialist heads. The downstream effect — pixel loss updating language-side representations — produces consistent perception gains instead of the usual multi-task regression.
- **Reasoning + perception interleaved.** `<|mask|>` inside `<think>` means a chain of thought can produce several masks mid-reasoning, each conditioned on everything thought so far. This is architecturally cleaner than running an external segmenter per candidate region.
- **Benchmark gap filled.** LensBench's cross-domain slices (aerial, driving, panoramic) and explicit/implicit axis target failure modes that RefCOCO underweights.

---

## Suggested follow-ups

- Deeper comparison with vision-token-grounded approaches (e.g. Molmo's point grounding) once Stage 1 recipe is published.
- Transfer of the mask-token design to larger backbones — the 4B result is clean, but scaling behavior is unknown.
- Combining `<|mask|>` with external tool use (segmenter-as-tool) — the paper's design does not preclude it.

---

## Sources

- Paper: *Xiaomi-Lens-0 Technical Report* — Xiaomi, 2026 — [arXiv:2610.23279](https://arxiv.org/abs/2610.23279) · [HF](https://huggingface.co/papers/2610.23279) · [AlphaXiv](https://www.alphaxiv.org/abs/2610.23279).
- Daily-papers digest: [2026-10-19 #7](../daily-papers/2026-10-19.md#7-xiaomi-lens-0).
