# Rubric-Based RL
*Depth — decomposing reference responses into atomic propositions and rewarding along multiple structured axes.*

**TL;DR:** Scalar outcome rewards can't distinguish "wrong final answer" from "hallucinated a specific fact"; both look like the same low score. Rubric-based RL decomposes each reference response into atomic propositions and scores generated answers per-component (e.g. Visual Faithfulness, Reasoning Consistency, Instruction Following), giving GRPO a dense, structured reward that localizes credit to specific spans and specific failure modes.

**Prereqs:** [grpo.md](grpo.md), [rlvr.md](rlvr.md), [_rewards.md](_rewards.md)
**Related:** [../post-training/reasoning/prm.md](../post-training/reasoning/prm.md), [cot-reward-model.md](cot-reward-model.md)

---

## What it is

A rubric-based reward is a **structured, multi-component scalar** derived from decomposing a reference response into atomic propositions and checking each independently. Introduced by V-Rubrics (Tian et al., 2026) for VLM post-training, the framework scores generated answers along three explicit axes:

- **Visual Faithfulness (VF)** — is each atomic proposition supported by the visual evidence?
- **Reasoning Consistency (RC)** — does the chain of reasoning validly follow from the visual facts?
- **Instruction Following (IF)** — are the user's constraints (format, scope) satisfied?

The reward is a weighted sum of per-component partial credit, and — when the supporting evidence span is known — credit is *prefix-localized* so the reward attaches to the span that carries the proposition rather than being smeared across the whole response.

## How it works

Pipeline:

1. **Rubric construction.** Reference responses are decomposed into atomic propositions using an LLM annotator (V-Rubrics used Gemini-3-Pro on a 50K-example training set derived from OpenMMReasoner sources).
2. **Per-example scoring.** Given a generated answer, each rubric item is checked; the aggregated per-component scores form the reward vector.
3. **GRPO with structured credit.** Instead of one scalar per rollout, GRPO consumes the component-wise reward with prefix localization — the policy gradient concentrates on the token spans responsible for missed rubric items.
4. **Training.** Starting from an SFT checkpoint (Qwen3-VL-8B fine-tuned on OpenMMReasoner-SFT-874K), V-Rubrics-GRPO beats both the SFT baseline and answer-only GRPO, with the largest gains on knowledge-oriented and visually grounded reasoning benchmarks.

## Why it matters

Rubric decomposition sits between two failure modes: rule verifiers can't score fuzzy outputs (there's no regex for "grounded in the image"), and preference RMs / VLM-as-judge scorers reward-hack easily. Rubric-based rewards give you dense, interpretable, per-component signal without training a reward model — the annotator work is offline and the scorer at RL time is a cheap check against pre-decomposed propositions. The technique is not VLM-specific: any task with reference responses that can be decomposed into atomic claims can be scored this way.

## Gotchas & tricks

- **Rubric quality caps everything.** LLM-annotated rubrics inherit the annotator's biases; V-Rubrics used a strong model (Gemini-3-Pro) and a structured prompt — cheaper annotators tend to produce vague, unrewardable rubric items.
- **Localization needs supporting-evidence spans.** Prefix-localized credit only works when the rubric annotation includes which part of the reference each proposition maps to. Without spans, the reward degrades to component-wise-but-diffuse credit.
- **Weighting matters.** VF/RC/IF weights control which failure mode the policy attacks first; equal weights are a fine default but different tasks (heavily visual vs heavily instruction-following) benefit from tuning.
- **Related to but distinct from PRM.** PRMs score self-generated reasoning steps; rubrics score reference-derived propositions. Rubrics are cheaper to annotate and less prone to reward hacking than dense PRM rewards used for RL.

## Sources

- Paper: *V-Rubrics: Visual Faithfulness via Rubric-Based Reinforcement Learning* — Tian et al., 2026 — [arXiv:2608.25580](https://arxiv.org/abs/2608.25580)
