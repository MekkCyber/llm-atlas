# Video Temporal Benchmarks
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Video-LLM temporal reasoning benchmarks have been vulnerable to language-only shortcuts: option wording, answer correlation, and language priors let models score without watching. TempCloze rebuilds the setup as a **video-cloze** task — given the beginning and end of a video, pick the true missing middle from four **same-source distractors** that share scenes and objects. Distractors probe three axes: **Semantic** (what event should happen), **Alignment** (when it should occur), **Progression** (how it should unfold). Across 31 Video-LLMs (10 proprietary + 21 open), the paper localizes the bottleneck: **Alignment** is the dominant failure mode.

**Prereqs:** [README](README.md)
**Related:** [../multimodal/README](../multimodal/README.md)

---

## What it is

A **video-cloze** benchmark: the model is shown a clip's opening and closing segments and must select the true middle from four candidate middles, all filmed from the same source video. Because distractors come from the same source, the *language of every candidate is nearly identical*; a text-only model can't win. This isolates the temporal-reasoning skill from the linguistic priors and multiple-choice heuristics that inflate scores on prior benchmarks.

TempCloze: 1,521 filtered videos, mainly long-take and egocentric — settings where temporal continuity is the harder inference.

## How it works

**Distractor axes** — each candidate middle is generated to fail along one axis while resembling the reference on the others:

- **Semantic distractor.** The event shown is *plausible* but wrong for what happens between the given beginning and end.
- **Alignment distractor.** The right event, wrong *time* — the middle would happen either before the shown beginning or after the shown end.
- **Progression distractor.** The right event, at the right time, but the *how* is scrambled — sub-actions in the wrong order.

**Same-source shared appearance.** All four candidates share the video's scene, subjects, and objects — the pixels look similar; the difference is *when* and *how*. This is what defeats language-only shortcuts (all candidates translate into similar English descriptions).

**Filtering.** Videos are curated for temporal specificity: sequences where the middle *must* satisfy geometric or physical continuity, not just any plausible content.

## Why it matters

- **Shortcut-robust evaluation.** Every prior long-video benchmark had a text-only baseline that scored above chance. TempCloze's same-source distractors collapse that shortcut.
- **Diagnostic decomposition.** Reporting *per-axis* accuracy tells developers which sub-skill is broken. The paper's headline result — Alignment is the bottleneck — is actionable.
- **Egocentric focus.** Long-take + egocentric videos are the target of upcoming agent + wearable use cases; that's where TempCloze concentrates.

## Gotchas & tricks

- **Distractor generation is expensive.** Same-source distractors require careful editing/sampling; the paper's filtered set of 1,521 is small — expect saturation once models specifically train against it.
- **Alignment axis is model-family sensitive.** Models with strong temporal positional encoding (RoPE-over-time, ALiBi-time) do better on Alignment than on Progression; use the per-axis split when comparing architectures.
- **Not a training signal.** The benchmark is eval-only; leaking it into training destroys its diagnostic value.
- **Language leakage residual.** Even same-source distractors can accidentally be describable by slightly different English. Audit the option texts before ranking a submission.
- **Report per-axis, not just aggregate.** An aggregate score can hide that a model is 90% on Semantic and 30% on Alignment.

## Sources

- Paper: *TempCloze: Can Video-LLMs Identify the Missing Middle?* — Pei, Zhao, Liu, Meng, Chen, Wang, Du (HKU and others), 2026 — [arXiv:2609.01515](https://arxiv.org/abs/2609.01515)
- Related: prior video-QA benchmarks (VideoMME, MVBench, EgoSchema) — the shortcut-vulnerable ancestors this benchmark targets.
