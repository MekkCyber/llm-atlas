# VibeVideoBench
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A 239-task bilingual (English + Chinese) benchmark asking multimodal agents to **write code that renders a complete MP4 video with sound inside one hour** — no calls to image or video generation models allowed. Evaluation combines program checks, ~13k rubric checks, and three VLM judge panels. Stresses agent planning, long-horizon tool use, and creative code output in a verifiable way.

**Prereqs:** *(none)*
**Related:** [../agents/README.md](../agents/README.md)

---

## What it is

Most multimodal benchmarks ask a model to *generate* a video directly. VibeVideoBench inverts the setup: the agent must author the full code pipeline — rendering (Manim, Three.js, Blender scripting, canvas), animation sequencing, audio mixing, encoding — that produces one. The deliverable is an MP4 file that must compile, run, and play.

Covers 12 task categories ranging from mathematical visualization and physics simulations to music video storyboarding, each with English and Chinese prompts.

## How it works

Three evaluation layers:

1. **Program checks.** Does the submitted code compile and run inside the one-hour budget? Is the output a playable MP4 of the required duration? Automatic, binary.
2. **Rubric checks (~13k across the benchmark).** Per-task, structured yes/no checks on specific elements (did the scene include a labeled axis? does the audio match the stated genre?). Scored by VLM judges against spec.
3. **Quality judgment.** Three VLM judge panels (GPT-6, Claude Opus 5.5, Gemini-class) rate the final video on fidelity to the prompt, visual quality, and audio quality. Panel rankings are reported with inter-judge agreement.

Benchmark reports both pass rate and judge-aggregated quality score per model. Human preference studies show judge rankings closely match human ones.

## Why it matters

- **Verifiable creative benchmark.** Produces a watchable artifact *and* a program that can be inspected and re-run — rare for creative tasks.
- **Surfaces delivery failures.** A major error class is agents that fail to finish inside the one-hour budget (unfinished renders, late submissions). This maps directly to real agent deployment constraints.
- **Code-driven creativity.** Tests whether agents can plan multi-step technical pipelines for a creative goal — a different skill from one-shot generation.
- **GPT-6 Astra and Claude Opus 5.5 lead** across nine agent systems tested. The gap over mid-tier systems is largely in planning and completion, not visual quality.

## Gotchas & tricks

- **Judge-model drift.** As judge models evolve, scores on the same submissions change. Freeze the judge version for cross-run comparability.
- **Rendering library choice.** Agents that know one visualization library well can appear worse than agents with broader toolkits; the benchmark isn't library-specific.
- **The one-hour budget bites hard.** Agents that spend 55 minutes designing and 5 rendering often miss the budget. Pacing is a real skill measured here.
- **Audio is a lever.** Several models skip audio entirely to save time; this is penalized by rubric checks on tasks that require it.

## Sources

- Paper: *VibeVideoBench: Evaluating Multimodal Agents on Code-Driven Video Generation* — Diao, Xu, Yang, et al., 2026 — [arXiv:2610.35864](https://arxiv.org/abs/2610.35864)
