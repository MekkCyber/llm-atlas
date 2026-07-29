# FilmBench
*Depth — a T2V and R2V benchmark grounded in professional Cinematic Language, with prompts reverse-engineered from award-winning film clips and an expert-grade automatic evaluator.*

**TL;DR:** A text-to-video (T2V) and reference-to-video (R2V) benchmark whose prompts are reverse-engineered from clips of award-winning films across 20 cinematic genres, evaluated against a professional Cinematic-Language taxonomy (**3 axes → 12 components → 35 T2V sub-metrics + 3 R2V-only**). Most prompts (1,056 of 1,169) are **multi-shot**. The automatic evaluator reproduces the human expert ranking at **Spearman ρ = 0.95 (T2V) / 0.96 (R2V)**.

**Prereqs:** [../evaluation/README.md](../evaluation/README.md), [../multimodal/README.md](../multimodal/README.md)
**Related:** [aime.md](aime.md)

---

## What it is

Prior video-generation benchmarks draw prompts from web sources or LLM templates and score with generic multimodal evaluators using rudimentary axes (overall quality, coarse text alignment, temporal smoothness). That measures basic video plausibility, not film-grade craft — where leading generators are still visibly deficient. FilmBench shifts the eval axis to what film school actually teaches.

## How it works

- **Prompt construction.** Every prompt is reverse-engineered from a clip of an award-winning film chosen by professional directors. Prompts follow real shot lists; **1,056 of 1,169 prompts are multi-shot**, contrasting with prior single-clip benchmarks.
- **Cinematic-Language taxonomy.** Evaluation splits into a three-level tree: 3 axes → 12 components → 35 T2V sub-metrics (+3 more for R2V). Axes cover cinematic craft dimensions (dynamic aesthetics, subject/scene, etc.) rather than generic video quality.
- **FilmOps.** An in-house expert-grade automatic evaluation agent, built around a suite of open-sourced Cinematic Language operators. Community can plug FilmOps operators into their own eval stacks.
- **Reference-to-video (R2V).** For R2V, a real reference video anchors the prompt, and 3 extra R2V-only metrics measure faithfulness to the reference.

## Why it matters

- **Headroom.** Leading T2V (9 evaluated) and R2V (7 evaluated) systems score well below their prior-benchmark numbers here — the benchmark still discriminates at the frontier.
- **Two crisp gap patterns.** Consistent under-performance on *dynamic aesthetics* across models, and a marked single-shot → multi-shot performance drop that widens for weaker models.
- **Automatic evaluator that agrees with humans.** ρ = 0.95 / 0.96 to expert rankings — good enough to use as a training-side reward proxy, not just a leaderboard.
- **Aligned with production use.** Real film work is multi-shot with cinematic-craft criteria; the benchmark reflects that.

## Gotchas & tricks

- Multi-shot metrics require the generator to actually produce multi-shot output; single-shot-only systems score effectively zero on those axes and drag down averages.
- FilmOps operators need calibration when applied to non-cinematic domains — they were built for film craft, not TikTok clips.
- Prompt reverse-engineering from real clips creates a copyright surface — the benchmark ships prompts, not the source clips.
- Human-ranking agreement is measured at the model level (ρ = 0.95/0.96); per-clip agreement is lower and should be checked before using FilmOps as a training reward.

## Sources

- Paper: *FilmBench: A Film-Grade Benchmark for Cinematic Video Generation* — Wang et al., 2026 — [arXiv:2607.24241](https://arxiv.org/abs/2607.24241)
