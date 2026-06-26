# Implicit Visual Chain-of-Thought (IV-CoT)
*Depth — split MLLM conditioning into a latent structural plan and a separate semantic stream for structure-aware text-to-image generation.*

**TL;DR:** Unified MLLM-based image generators struggle to preserve object counts, spatial relations, and attribute bindings because *structural planning* and *appearance rendering* are entangled in a single conditioning stream. IV-CoT (Li et al., 2026) decomposes the conditioning into two query groups inside one transformer: *structural queries* form a latent visual plan (trained-only sketch supervision), *semantic queries* then render appearance on top of that plan — all in a single forward pass, no explicit sketch decoding at inference.

**Prereqs:** [attention](../fundamentals/attention.md), [multi-head-attention](../architectures/multi-head-attention.md)
**Related:** [_data-curation](../data/_data-curation.md), [post-training/reasoning/long-cot-rl](../post-training/reasoning/long-cot-rl.md)

---

## What it is

A conditioning-side modification for unified MLLM text-to-image models. The input set of conditioning tokens is partitioned into two named groups, both used at every cross-attention layer of the image decoder:

- **Structural queries.** Trained to encode layout (object count, position, coarse shape) — supervised at training time only by a sketch-reconstruction loss.
- **Semantic queries.** Trained to encode appearance (texture, color, identity) — supervised by the standard image-generation loss.

At inference, both groups are used jointly; the sketch is *never* decoded. The model performs implicit chain-of-thought via the structural queries' latent plan.

## How it works

```
prompt → unified MLLM encoder → conditioning tokens Z
Z is split: Z_struct (k tokens), Z_sem (n tokens)

— training only —
Z_struct → small sketch head → predict ground-truth sketch image
                                  ↑ supervises Z_struct to capture layout

— always —
image decoder cross-attends to [Z_struct, Z_sem]
loss = standard diffusion loss on RGB image
```

Key design choices:

1. **Cascade, not concat.** The model is encouraged to use $Z_{\text{struct}}$ as a *plan* that $Z_{\text{sem}}$ conditions on. Training tricks (sketch-only ablations, alternating losses) enforce that the structural queries genuinely encode layout rather than texture.
2. **No intermediate decoding at inference.** Sketch supervision is training-only — the structural queries function as a latent plan.
3. **Single forward pass.** Unlike "plan then generate" or "generate then refine" pipelines, IV-CoT generates in one shot.

## Why it matters

- **Structure-aware T2I gains.** Improvements on GenEval (object count, spatial relations, attribute bindings) and T2I-CompBench, without the inference-time cost of two-pass generation.
- **Cleaner inductive bias for "reasoning for generation."** Most prior work either threads a separate text plan through the model (slow, brittle) or generates then reranks (wasteful). IV-CoT bakes the plan into the conditioning stream and supervises it indirectly.
- **Generalizes beyond images.** The same structural/semantic split is plausibly applicable to video generation (where layout matters even more), 3D generation, and structured-document synthesis.

## Gotchas & tricks

- **Sketch supervision must be cheap.** A heavy sketch decoder during training adds non-trivial compute; the paper uses a small head.
- **Balance the two losses.** Too much sketch weight collapses semantic queries into duplicates; too little fails to differentiate the streams. Treat the sketch-loss weight as a hyperparameter.
- **Sketch ground truth.** Use canny edges or off-the-shelf sketch extractors — exact target style doesn't matter much; structure does.
- **Visualization sanity check.** After training, visualize structural-query attention maps: they should cluster by spatial region, while semantic-query maps should cluster by object identity / texture.
- **Doesn't replace post-training.** IV-CoT is a *training-time* architecture change; it's complementary to reward-model finetuning and RLAIF for image generators.

## Sources

- Paper: *IV-CoT: Implicit Visual Chain-of-Thought for Structure-Aware Text-to-Image Generation* — Li, Lin, Xiao, et al. (NLPR/CAS, Ant Group, HKU), 2026 — [arXiv 2606.24849](https://arxiv.org/abs/2606.24849).
- Benchmarks: GenEval (Ghosh et al., 2023), T2I-CompBench (Huang et al., 2023).
