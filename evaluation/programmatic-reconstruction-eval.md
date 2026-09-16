# Programmatic Reconstruction Evaluation
*Depth — an evaluation protocol that asks an agent to reconstruct real-world content by writing code, then scores the reconstruction on both factual retention and perceptual similarity to the source.*

**TL;DR:** Traditional multimodal benchmarks probe understanding with question answering. Programmatic reconstruction evaluations flip this: if an agent truly understands a video (or image, or 3D scene), it should be able to **write code that recreates it** in a shared authoring tool. Scoring is a joint over **factual retention** (spatiotemporal facts preserved) and **perceptual similarity** (how close the rendered reconstruction looks to the source). Introduced by BVB (Blender-VideoBench) in 2026 for agentic video understanding.

**Prereqs:** (none)
**Related:** [humaneval](humaneval.md), [mmlu](mmlu.md)

---

## What it is

For a modality where the space of possible answers is uncountable — video, 3D scenes, complex images — QA-style evaluation is fragile and games easily. Programmatic reconstruction defines the correctness signal differently: the agent produces a **program** in a shared authoring environment (Blender, in the BVB instantiation), the program is executed to render an output, and the output is compared to the source along two orthogonal axes:

1. **Semantic axis (Dual VQA)** — a QA-style measure of how many spatiotemporal facts the reconstruction preserves. Both source and reconstruction are queried, and matching answers score.
2. **Perceptual axis (Latent Similarity)** — a learned similarity between the rendered reconstruction and the source video, calibrated in a blind human-preference study.

The final score is a square-root mean over the two, penalizing unbalanced performance.

## How it works

- **Shared harness.** Every agent programs through the same lightweight harness (Mini-BVB) in an identical sandbox under a shared cost limit — a fair-comparison mechanism so model differences dominate rather than tooling differences.
- **Same camera.** Reconstruction is rendered from the source's camera trajectory so alignment between source and reconstruction is well-defined.
- **Dual metric.**
  - *Dual VQA:* a fixed set of spatiotemporal questions is asked of both the source and the reconstruction; matching answers count.
  - *Latent Similarity:* a learned perceptual similarity, validated by a blind human-preference study (15 raters, 5 configurations in the BVB report), correlates with human judgment.
- **Cost budget.** Each configuration operates under the same total-cost cap on tokens / tool calls / rendering. Under-spending is not rewarded; over-spending is impossible.

## Why it matters

- **QA-only benchmarks are approaching saturation and are gameable.** Programmatic reconstruction is a much harder ability test — the agent has to *build* something that reproduces the source, not just describe it.
- **Correlates with human preference.** Latent Similarity's strong correlation with human raters in the BVB study makes the perceptual axis defensible.
- **Exposes a real failure mode.** In the BVB report, the best system reaches Latent Similarity **88.6** but retains only **53.7%** of the source-correct spatiotemporal answers, and more reasoning does not close the factual gap. That divergence names a research problem the community would otherwise have missed.
- **Generalizable.** The protocol shape (shared harness → reconstruction program → dual-axis score) transfers to images (SVG/vector), 3D scenes (any procedural scene format), audio (score/DSL), and physical setups (URDF).

## Gotchas & tricks

- **Latent similarity is an approximation.** The perceptual score is a *learned* similarity, so it inherits the biases of its training. Blind human validation on each new instantiation is not optional.
- **Sandbox specifics leak into scores.** Different Blender versions produce different renderings for the same script; pin versions, seeds, and render settings.
- **Cost budget is the equalizer.** Without a shared cost cap, richer agents win by trying more configurations and picking the best; the shared budget forces one-shot competence.
- **Semantic retention is the current ceiling.** The failure mode BVB names (visual similarity high, factual retention low) suggests reconstructions that "look right" from a distance but get details wrong. Metrics that only look at similarity will miss this.

## Sources

- Paper: *BVB: Benchmarking Agentic Video Understanding via Programmatic Reconstruction in Blender* — Tang, Shimada, Meng, Bi, Liu, et al. — 2026 — [arXiv:2609.15478](https://arxiv.org/abs/2609.15478).
