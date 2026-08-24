# Chain-of-Experience
*Depth — inference-time framework that lets an LLM accumulate feedback traces across questions to improve continually without retraining.*

**TL;DR:** Standard LLM eval treats each question as independent. Chain-of-Experience (CoE) threads curated summaries of prior questions, feedback signals, and past attempts into the context of each new question, so the model learns from its own recent history without any weight update. Across 8 frontier models (GPT-5, Gemini-2.5 Pro, Claude-4.5 Sonnet, ...) on math, code, and knowledge domains: **+5.6% overall accuracy, -19% API cost**. Most of the win lands early.

**Prereqs:** [_post-training.md](_post-training.md), [reasoning/long-cot-rl.md](reasoning/long-cot-rl.md)
**Related:** [../agents/skill-workflow-coevolution.md](../agents/skill-workflow-coevolution.md), [../evaluation/memtrap-bench.md](../evaluation/memtrap-bench.md), [rejection-sampling.md](rejection-sampling.md)

---

## What it is

An inference-only continual-learning wrapper. No weights are updated; instead, a structured *experience trace* is maintained alongside the model:

- **Solved-item summaries** — compact notes on what worked for prior questions.
- **Feedback channels** — self-reflection, external verifier signals, ground-truth labels when available.
- **Failure traces** — where the model went wrong before and why.

Each new question is prepended with the subset of the trace most relevant to it (selected by embedding, task-type tag, or heuristic).

## How it works

The core loop:

```
experience = []
for q in stream:
    context = select_relevant(experience, q)     # retrieve past traces
    answer = model(q, context)
    feedback = collect_feedback(q, answer)       # self / external / ground-truth
    experience.append(summarize(q, answer, feedback))
```

Three design levers:

1. **Multi-channel feedback.** Self-reflection ("was I right? why?"), external tool signals (compiler errors, unit tests, verifiers), and ground-truth labels each contribute distinct information. The paper shows they combine additively — no single channel is enough.
2. **Trace curation.** Raw prior contexts blow the context window and hurt more than they help. Summarize each experience into a short structured note (task type, failure mode, remedy). Retrieval over the notes, not the raw traces.
3. **Selection budget.** Include only the top-k most relevant notes for the new question. Empirically, small k (~3–5) beats larger k on both accuracy and cost.

The gains concentrate in the first few iterations: the model quickly patches its most systematic failure modes; subsequent gains taper.

## Why it matters

An inference-only complement to training-time continual learning. Cheap to deploy — no gradient updates, no new checkpoint — and applicable to closed-weight models where fine-tuning is not an option. Also gives an early signal on which failure modes are *quickly patchable* vs *stubbornly baked in*: modes that CoE fixes are candidates for cheap in-context mitigation; modes that resist need post-training.

The multi-channel-feedback ablation is a general insight: any continual-improvement pipeline (train-time or in-context) benefits from stacking heterogeneous signals rather than perfecting one.

## Gotchas & tricks

- **Watch context-window budget.** Naive concatenation of prior experience saturates the window before useful examples fit. Aggressive summarization is not optional.
- **Feedback quality dominates.** Self-reflection alone is prone to the model reinforcing its own bugs. Prefer external signals (verifier, compiler) whenever available, with self-reflection as a backup.
- **Selection matters more than volume.** Including irrelevant experience is a distractor attack the model runs on itself. Retrieval scores matter.
- **Overfitting to the recent stream.** If the question distribution shifts, cached experience can misdirect the model. Age-weight the trace.
- **Diminishing returns are real.** The first 3–5 iterations do most of the work. Long-lived CoE runs need periodic pruning to avoid trace bloat.
- **Distinct from RAG.** CoE traces are *model-authored* records of solved-question experience; RAG documents are *externally-authored* knowledge. Both can coexist.

## Sources

- Paper: *Chain-of-Experience for Continual LLM Improvement* — Tu, Fang, Wang, Xie, Yan, UCSC/ByteDance, 2026 — [arXiv:2608.18027](https://arxiv.org/abs/2608.18027).
- Related: *Self-Refine* (Madaan et al., 2023) — single-item self-reflection; CoE generalizes across items.
- Related: *Reflexion* (Shinn et al., 2023) — episodic reflection with a memory store; CoE differs on the multi-channel feedback and structured trace curation.
