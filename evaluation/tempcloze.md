# TempCloze
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A **video cloze** benchmark for temporal reasoning in Video-LLMs. Given a video's beginning and ending clips, the model must pick the true missing middle from four **same-source distractors** that share scenes and objects with the correct answer. 1,521 filtered long-take / egocentric videos; distractors are engineered along three axes (Semantic, Alignment, Progression). Across 10 proprietary + 21 open-source Video-LLMs, **Alignment** (knowing *when* an event should occur) is the primary bottleneck.

**Prereqs:** [README.md](./README.md), [../multimodal/README.md](../multimodal/README.md)
**Related:** none

---

## What it is

Standard Video-LLM temporal-reasoning benchmarks pose a question and a set of MCQ options phrased in language. That structure lets a strong LLM exploit option wording, answer correlations, or language priors to score correctly *without watching the video*. TempCloze plugs the leak two ways:

- **Cloze framing.** Show the model the *beginning* and *ending* clips of a video and ask it to pick the true *middle* from four candidate clips. The task is visual identification, not textual QA.
- **Same-source distractors.** All four candidates come from the same source video and share scenes, objects, and lighting; the differences are strictly temporal. Distractors are constructed along three explicit axes:
  - **Semantic** — what should happen next.
  - **Alignment** — when the event should occur relative to the beginning and end.
  - **Progression** — how the event unfolds internally.

## How it works

Dataset: 1,521 videos, mostly long-take and egocentric, filtered from seven sources. Each video is split into (beginning clip, middle clip = ground truth, ending clip); three same-source distractor middles are generated for each of the three axes.

Evaluation: for each video, present the model with the beginning + ending clips and four candidate middles; the model outputs a choice. Report accuracy overall and per-axis.

Because distractors share appearance with the correct answer, appearance-only VLMs collapse to chance on the Alignment axis but retain some ability on Semantic (where the semantic content of the middle differs qualitatively).

## Why it matters

- **Isolates alignment as the current frontier bottleneck.** Across 31 evaluated Video-LLMs, Semantic and Progression are handled reasonably, but Alignment is where all systems degrade — evidence that Video-LLMs can identify *what* is happening but not *when* it should happen relative to context.
- **Cloze pattern generalizes.** The same "same-source distractors + cloze reconstruction" pattern is applicable to any temporal-reasoning benchmark (audio-LLM, sequential-image reasoning, code-execution traces).
- **Clean leaderboard structure.** Reporting per-axis accuracy gives researchers a targeted signal: a system that improves overall accuracy but not on Alignment is not making progress on the interesting axis.

## Gotchas & tricks

- **Long-take / egocentric bias.** The video mix leans egocentric; models trained heavily on third-person scripted video may under-perform for reasons unrelated to temporal reasoning.
- **Distractor construction quality dominates.** Weak distractors let the model succeed via appearance shortcuts even in cloze framing; the paper's filtering pipeline (7 sources, careful same-source constraints) is essential to reproduce.
- **Frame-count budget matters.** Video-LLMs subsample frames; the alignment axis in particular is sensitive to how many frames the model sees per clip. Report the frame budget alongside the accuracy.
- **Not a substitute for temporal QA benchmarks.** TempCloze isolates one axis (visual temporal alignment) cleanly; downstream tasks that require language-grounded temporal reasoning (e.g., "when did the person open the drawer?") remain in scope for text-QA video benchmarks.

## Sources

- Paper: *TempCloze: Can Video-LLMs Identify the Missing Middle?* — Pei, Zhao, Liu, Meng, Chen, Wang, Du (University of Hong Kong), 2026 — arXiv:2609.01515.
