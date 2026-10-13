# VisionWeave
*Depth — elastic visual-token routing for MLLMs.*

**TL;DR:** MLLMs spend compute on a fixed, dense budget of visual tokens even where images carry little information. VisionWeave produces *two* streams per image — a native fine-grained stream and a coarse stream from a gated spatial pooler — and learns a **granularity router** that decides, per spatial region, which stream flows into the LLM. Trained via self-distillation. Saves 43% of tokens while retaining 98.9% of native performance, 2.3× end-to-end throughput on SGLang.

**Prereqs:** [attention](../fundamentals/attention.md), [_moe](../architectures/_moe.md)
**Related:** [capacity-factor](../architectures/capacity-factor.md)

---

## What it is

An architectural modification for a multimodal LLM's visual encoder path. Where standard pipelines commit to one visual-token resolution per image, VisionWeave produces a fine and a coarse representation side by side and lets a learned router pick which representation each region of the image contributes to the LM. The router is a small MoE-shaped head over the spatial grid, so routing is MoE *over granularities*, not over experts.

## How it works

1. **Gated spatial pooler.** A learned pooling layer compresses patches of the native visual-token grid into coarse tokens. The gate is conditioned on local image content.
2. **Granularity router.** For each spatial region, a small router picks between native-fine and pooled-coarse tokens. Picks are deterministic at inference (argmax) and softmax-weighted during training.
3. **Self-distillation training.** The routed model is trained to match the full-token "teacher" on paired inputs. Loss combines next-token prediction with an alignment term on hidden states where the routed and teacher streams coincide.

Serving: on SGLang, the router runs once; downstream LLM layers see a smaller token budget, so KV-cache and attention cost drop proportionally.

## Why it matters

Elastic token budgets per image are structurally correct for video and document-heavy workloads where information is unevenly distributed. The routing-over-granularities pattern is close to a *vision-side MoE* and generalizes to video (coarse temporal pooling + fine spatial) and multi-page docs. It also degrades gracefully: if the router under-prunes, you still get a correct answer, just more compute.

## Gotchas & tricks

- **Router warm-up.** Early in training the router has no signal and tends to collapse to all-fine (safe). Add a mild entropy bonus or a token-saving auxiliary loss during warm-up.
- **Benchmark coverage.** 50% fixed-pruning baselines retain ~88%; VisionWeave retains ~98.9% at ~57% budget. The win is largest on image-rich benchmarks (charts, documents) and smallest on text-dominated ones.
- **Not useful without a visual encoder with spatial structure.** Methods that produce a flat 1D token stream (ViT with global pooling) can't be routed this way.

## Sources

- Paper: *VisionWeave: Weaving Elastic Visual Representations as a Native Capability of MLLMs* — Feng, Yang, Chen et al., 2026 — [arXiv:2610.07987](https://arxiv.org/abs/2610.07987)
