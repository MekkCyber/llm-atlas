# GST-Bench

*Depth — a video benchmark that grades whether VLMs build a persistent 3D scene understanding from egocentric video, not just per-frame retrieval.*

**TL;DR:** VLM video benchmarks mostly test whether models can identify or describe things visible in the input frames. GST-Bench (2026) targets a different capability: **global spatial awareness**. It asks VLMs to answer questions from *viewpoints that never appeared* in the input video, and to convert egocentric observations into *top-down* representations. 2,762 human-verified questions over 6,790 minutes of synthetic video. The strongest zero-shot VLM gets 42.68 vs. human 79.08 — a large, standing gap.

**Prereqs:** *(basic VLM familiarity)*
**Related:** [README.md](README.md), [../multimodal/README.md](../multimodal/README.md)

---

## What it is

A benchmark of **novel-view spatial questions** on synthetic egocentric video, with two headline task shapes:

- **Unseen-viewpoint QA.** Given a video, answer a question about the scene as it would appear from a viewpoint *not present* in any frame. Forces reasoning over an internally maintained 3D scene, not retrieval.
- **Egocentric → top-down conversion.** Produce a top-down (bird's-eye) description or grounding from purely first-person video. Directly probes whether the model built a consistent global spatial map.

Video is synthetic (renderer-controlled), so ground truth is exact. Questions are human-verified.

---

## How it works

**Data construction.** Rendered scenes give the benchmark access to true 3D layouts; questions are generated to target novel viewpoints and top-down references, then human-verified for solvability from the input video alone.

**Task format.** Multiple-choice or short-answer format; the judge is deterministic given the reference answer.

**Metrics.** Accuracy over the full 2,762-question set, plus splits by task type (unseen-viewpoint vs. top-down) and by scene complexity.

**Findings.** Best zero-shot VLM: 42.68 (vs. human 79.08 baseline). Failure-mode analysis: proprietary models fail at cross-frame integration; open-source models are limited both in perception and reasoning. In short — nothing shipped in 2026 actually builds the persistent scene model the benchmark demands.

## Why it matters

- **Isolates a capability the field is quietly assuming.** "Agentic vision" systems (navigation agents, robotics VLMs) all *assume* their base VLM has a coherent spatial model. GST-Bench provides direct evidence that most do not.
- **Novel-view framing is hard to game.** Object-count or grounding benchmarks can be solved by per-frame recognition + heuristics; unseen-viewpoint questions cannot.
- **Standing target for training.** Because the gap to human is so large (36+ points), models can improve on GST-Bench for several years before saturating.

## Gotchas & tricks

- **Synthetic-only.** Ground truth is exact but the distribution is renderer-shaped; real egocentric video adds noise (motion blur, exposure) not represented here. Complement with real-video benchmarks before claiming general spatial awareness.
- **Multiple-choice inflation.** As with all MCQ benchmarks, chance-correct answers inflate small-model scores; watch the calibration curve and free-form variants when available.
- **Top-down conversion depends on the output format.** A model that "understands" the top-down layout but can't emit the expected string format scores low. Include a lenient string-similarity fallback for out-of-format outputs when comparing candidates.
- **Cross-frame integration is the bottleneck.** Failure mode analysis points at temporal integration — pair with existing multi-frame handling patches (memory tokens, video tokenizers) rather than perception-only fixes.

## Sources

- Paper: *GST-Bench: Can VLMs Develop Global Spatial Awareness from Video?* — Huang et al., ByteDance Seed / Zhejiang U. / NUS, 2026 — [arXiv:2608.05747](https://arxiv.org/abs/2608.05747).
