# SemComp-Bench
*Depth — an outcome-oriented evaluation for text-to-video generation.*

**TL;DR:** SemComp-Bench reframes video-generation evaluation from perceptual quality to **semantic task completion**: given a text instruction and a reference image, does the generated video actually achieve the requested outcome while staying grounded in the reference? Tu et al. (2026) contribute a six-domain dataset (SemComp-Data), a VLM-as-judge protocol using binary outcome questions, and two axis scores — **Outcome Achievement** and **Generation Reliability**. Reveals a gap perceptual metrics hide: SOTA video generators trade completion off against grounding.

**Prereqs:** *(none)*
**Related:** [../multimodal/README.md](../multimodal/README.md), [ifeval.md](./ifeval.md)

---

## What it is

Paper: *SemComp-Bench: Benchmarking Semantic Task Completion in Video Generation*, Tu et al., 2026, arXiv 2608.17426.

- **Task shape.** Given a reference image + text instruction, generate a video whose *outcome* matches the instruction while the *cause* remains semantically grounded in the reference.
- **Dataset (SemComp-Data).** Six domains covering object manipulation, agent action, scene evolution, physical interaction, spatial navigation, and abstract transformation. Each item carries a reference image, an instruction, and outcome-focused ground-truth clips.
- **Evaluation.** A vision-language model reads the generated clip and answers a bank of outcome-focused **binary questions** ("does the object end in the specified state?", "does the actor perform the specified action?"). Aggregating gives two scores per model:
  - **Outcome Achievement (OA)** — was the requested outcome achieved?
  - **Generation Reliability (GR)** — was it achieved for the *right* reasons (grounded in the reference)?
- **Reporting.** Both scores per domain, plus an OA/GR frontier plot.

## How it works as an LLM/VLM eval

- Judge model runs offline; each generated video is scored against a fixed VLM (paper reports agreement with human judgement).
- Binary questions are more robust than free-form judgement — reduces judge-variance across VLMs.
- Two-axis scoring exposes the *cheating* mode where a generator hits OA by ignoring the reference image entirely.

## Why it matters

- **Fidelity ≠ correctness.** Perceptual metrics (FVD, CLIP-Score) reward pretty video, not the right video. SemComp-Bench separates the two.
- **Two-axis view breaks the ceiling.** Modern generators score well on either OA or GR but rarely both — the frontier plot shows the tradeoff explicitly, which single-scalar metrics collapse.
- **Outcome orientation is the natural next objective.** After the Sora/VEO/Runway quality races, "does the video complete the task?" is the question a downstream user actually asks.

## Gotchas & tricks

- **Judge VLM matters.** Absolute OA/GR numbers depend on the judge; the paper reports human agreement but treat cross-paper score comparisons with care unless the judge matches.
- **Binary-question coverage is the ceiling.** If the question bank misses an outcome dimension, models can cheat that dimension for free.
- **GR is harder to game than OA.** Practitioners tuning against SemComp-Bench should target the two axes together — hill-climbing OA alone rewards ungrounded generation.
- **Reference-image bias.** Some domains have narrow visual prior distributions; a generator that memorised the reference distribution can inflate GR without genuine grounding.
- **Not a physics benchmark.** Outcome achievement is judged on VLM-visible outcomes, not physical plausibility — a generator that "achieves" an outcome via a visually plausible but physically impossible clip still scores well.

## Sources

- Paper: *SemComp-Bench: Benchmarking Semantic Task Completion in Video Generation* — Tu et al., USTC / DeepGlint, 2026 — [arXiv 2608.17426](https://arxiv.org/abs/2608.17426) — introduces the dataset, the OA/GR protocol, and the frontier study.
