# SpatialGen-Bench (ProVisE)
*Depth — spatial-cognition evaluation that lets image-generation models answer in pixels, not text, and parses those pixels back into the original benchmark's metric.*

**TL;DR:** Existing spatial-reasoning benchmarks (depth, occupancy, layout, path) require text or coordinate answers, which forces image-generation models to translate rather than reason spatially — an *answer-interface mismatch*. **ProVisE** (Protocolized Visual Evaluation) constrains the model's output to inspectable-pixel formats (depth fields, masks, drawn paths) that a parser converts back to the benchmark's original scoring form. **SpatialGen-Bench** ships 470 curated samples across 14 spatial subtasks organised into four capability levels, plus an agentic protocol-construction step that transfers ProVisE to six external benchmarks.

**Prereqs:** [../evaluation/README](README.md), [../multimodal/README](../multimodal/README.md)
**Related:** [mmlu](mmlu.md)

---

## What it is

An evaluation harness where the answer *interface* is chosen to match the model class:

- **Text-output VLMs** answer with strings/coordinates as before.
- **Image-generation models** answer with a *visual answer* — for instance, output a depth map, an occupancy mask, or an image with a highlighted region — that a parser converts into the same numerical form the text-output benchmark expects.

SpatialGen-Bench organises 470 samples into 14 subtasks across four capability levels (from primitive spatial perception up to compositional reasoning). Every task is defined so that both text and pixel answer formats are meaningful, enabling a like-for-like comparison across model classes.

## How it works

The core insight: pixel-answers are lossless for many spatial questions because the answer *is* a spatial artefact. Forcing text output introduces a translation step that (i) is not what the model was trained to do, (ii) creates its own error mode, and (iii) confounds spatial cognition with textual expressiveness.

ProVisE decouples the two:

1. **Task definition.** Fix the spatial question and its ground truth (a depth map, a mask, a path).
2. **Constrained visual output.** The image model produces a bounded output whose semantics are pre-agreed (colour = depth in metres, mask channel = the answer region).
3. **Parser.** Deterministic post-processing converts the pixels to the metric's numerical input (e.g. average absolute depth error, IoU of the mask against ground truth).

The paper also introduces an **agentic protocol-construction** step that generates ProVisE-compatible protocols for six pre-existing external benchmarks — showing the interface-matching idea generalizes.

## Why it matters

- **Cleanly separates representation from expression.** "Does the model represent space?" is a different question from "can it narrate space?" ProVisE lets a benchmark answer only the first.
- **Reveals complementary strengths.** Image-generation models excel on inspectable-pixel answers; text-output VLMs win on compositional transformations that require going beyond visible evidence. Neither class dominates.
- **Puts image generators on the leaderboard.** Prior benchmarks effectively excluded them by answer format; ProVisE fixes that.

## Gotchas & tricks

- **Parser design is load-bearing.** A brittle parser silently inflates the image-generator's error rate. Prefer parsers that accept a range of valid renderings (colour tolerance, mask fuzziness).
- **Not every task fits.** Compositional spatial reasoning that requires unseen transformations still favours text output; ProVisE augments rather than replaces text-based spatial eval.
- **Ground-truth format has to be visual too.** If ground truth is a coordinate, define its visual analogue (e.g. a point marker in an image) explicitly, or the interface remains mismatched.
- **Transfer step is agentic.** The six-external-benchmark transfer uses an agent to construct ProVisE-style protocols; check the generated protocols for faithfulness to the original tasks.

## Sources

- Paper: *Show, Don't Tell: Evaluating Spatial Cognition in Generative Pixels Rather Than LLM Text*, 2026 — [arXiv:2607.21072](https://arxiv.org/abs/2607.21072).
