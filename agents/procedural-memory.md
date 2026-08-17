# Procedural memory

*Depth — distill verified rollouts into reusable natural-language "lessons," score each by transfer reliability, and retrieve them at inference to guide a frozen model.*

**TL;DR:** Procedural memory is a parameter-update-free way to add narrow skills to a frozen agent: at exploration time it converts successful (verifier-checked) rollouts into short, transferable "lessons," attaches a **Transfer Reliability Score (TRS)** to each lesson that is calibrated from later retrieval outcomes, and at deployment retrieves the most relevant lessons and injects them into the prompt. The Spatial Memory Agent (SMA, 2026) is the canonical instantiation: it beats every base VLM it's stacked on across five spatial benchmarks with **no weight update** and **no inference-time tool call**.

**Prereqs:** [_harness-optimization](_harness-optimization.md), [rejection-sampling](../post-training/rejection-sampling.md)
**Related:** [agent-skills](agent-skills.md), [darwinx](darwinx.md)

---

## What it is

An external, retrieval-first memory bank of *procedural* content — lessons about *how to solve* a class of problems, not facts. Each entry is:

- **Lesson text** — a short natural-language rule distilled from a verified rollout ("when the target is out of view, first sweep left before turning right").
- **Trigger conditions** — semantic descriptors used at retrieval time (setting, task type, entity hints).
- **Transfer Reliability Score (TRS)** — a scalar that starts uniform and is updated after each retrieval based on whether the retrieved lesson helped the downstream task.

The bank is written to during exploration (with a verifier in the loop) and read from at deployment (no writes, no gradients, no tool calls).

## How it works

**Write path (offline / exploration).** The frozen VLM is queried against a verifiable spatial environment. Rollouts are scored; wins and instructive losses are handed to a **reflection prompt** that extracts a compact transferable lesson. Each new lesson enters the bank with TRS initialized to the average.

**Score-calibration path.** After each future retrieval, whether the retrieved lesson led to a correct answer is used to update the lesson's TRS. High-TRS lessons rise in the ranking; consistently unhelpful lessons decay to the tail.

**Read path (deployment).** For each new query, a two-stage retrieval fires:

1. **Semantic filter** narrows the bank to lessons whose trigger conditions match the query.
2. **Similarity × TRS ranking** picks the top-k of those to inject into the model prompt.

Retrieved lessons enter as bullet points in a "notes from past experience" block that the frozen model can attend to when producing its next answer.

## Why it matters

- **No fine-tuning, no expert tools at inference.** All standard alternatives for spatial reasoning either fine-tune (SFT/RL cost) or call heavy external tools (depth estimators, 3D reconstruction). Procedural memory needs neither.
- **Composes with everything.** Layers cleanly on top of an already-fine-tuned agent or an already-tool-using harness — it changes only the prompt.
- **Concrete gains.** Across 5 spatial benchmarks × 4 base VLMs (20 evaluations), SMA has the highest macro-average per base model and wins the majority of individual settings — with a frozen model.

## Gotchas & tricks

- **Retrieval hit rate is the whole story.** If similarity retrieval misses the applicable lesson, the model just runs as if the bank weren't there.
- **Ceiling = base model.** Procedural memory cannot exceed what the model can execute once the lesson is in-prompt. Test with the lesson literally inlined; if the model still fails, memory won't save you.
- **TRS drift.** A lesson can win a streak by luck and rise; add a shrinkage prior or bootstrap CI around TRS so a few lucky retrievals don't lock a bad lesson at the top.
- **Bank rot.** Distilled lessons age when the environment or model changes. Time-decay TRS or periodically re-verify sampled lessons.
- **Prompt bloat.** Injecting too many top-k lessons kills the model's attention over the actual task. The paper keeps top-k small (single-digit) and evaluates it.

## Sources

- Paper: *Spatial Memory Agent: Experience-Grounded Procedure Memory for Spatial Intelligence* — Zhang, Ding, Zhou, Du, Zhang, Zhao, Xi, Chen, 2026, [arXiv:2608.12743](https://arxiv.org/abs/2608.12743)
