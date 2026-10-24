# Structured Next-Token Perception
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Cast every visual-perception output — bounding boxes, counts, OCR strings, referring expressions, segmentation masks, matting alphas — as a token sequence emitted by a vision-language model. A dedicated `<OBJ>` token hands off to a small pixel decoder when the task is pixel-level; the LLM itself stays autoregressive. The result is one unified VLM that beats task-specific heads across 30+ perception benchmarks (WeVisOne-VL, Tencent 2026).

**Prereqs:** [../fundamentals/attention.md](../fundamentals/attention.md), [../fundamentals/_tokenization.md](../fundamentals/_tokenization.md)
**Related:** [README.md](README.md)

---

## What it is

A unified perception architecture for VLMs. Instead of attaching per-task heads (detection head, segmentation head, OCR head, …), every perception output is serialised into tokens with a task-shared schema. Object-level outputs (coords, labels, numbers, strings) are plain tokens; pixel-level outputs are triggered by a special `<OBJ>` token that hands off to a lightweight pixel decoder reading from the LLM's hidden state at that position.

## How it works

- **Schema.** Each perception task is defined by a token grammar: for detection, `<cls> x y w h ;` repeated; for OCR, raw text; for counting, a digit; for segmentation, `<OBJ>` followed by an identifier that selects which object to mask.
- **Pixel decoder.** Small transformer (described as "lightweight" in the paper) that takes the hidden state at an `<OBJ>` token position plus a learned query bank and produces a dense mask or alpha matte. The LLM does not generate pixel values; the decoder does.
- **Training.** Multi-stage curriculum: first object-level tasks only, then pixel-level with `<OBJ>` wiring, then full joint. Pixel-level loss is standard mask/IoU; object-level is cross-entropy on the token stream.
- **Inference.** Standard autoregressive decoding until `<OBJ>` fires, at which point the decoder is run inline using the current hidden state. The LLM resumes decoding on the next step.

## Why it matters

- **One checkpoint, many tasks.** Replaces a stack of task-specific perception models with a single VLM that still beats specialists on 30+ benchmarks.
- **Pixel outputs without a second branch.** Earlier unified VLMs either skipped segmentation/matting or added a parallel generative branch (e.g. diffusion-side). The `<OBJ>`-token bridge is cheaper and keeps the LLM purely autoregressive.
- **Composable with LLM tooling.** Because all outputs are tokens, standard LLM infra — streaming, structured decoding, tool calling — applies out of the box, including for pixel-level tasks.

## Gotchas & tricks

- **Grammar drift.** Free-form LLM decoding can produce malformed coord sequences; the paper constrains decoding with grammar-aware masking for structured outputs.
- **`<OBJ>` placement matters.** Firing the token too early produces masks for underspecified referents. Training must teach the model *when* to emit it, not just how to mask once emitted.
- **Mask quality is decoder-bound.** The pixel decoder is small by design; segmentation quality on long-tail categories is likely limited by the decoder's capacity, not the LLM's.
- **Benchmark coverage ≠ zero-shot generalisation.** Beating specialists on evaluated tasks doesn't mean the schema generalises to unseen task families — the token grammar has to be defined.

## Sources

- Paper: *WeVisOne-VL: Unified Visual Perception from Objects to Pixels* — Liang et al., Tencent, 2026 — [arXiv:2610.31329](https://arxiv.org/abs/2610.31329).
