# VLM-Judge Reward (Chunk-and-Vote)
*Depth — pairwise VLM reward for long-horizon multimodal generation, aggregated by voting over action-aligned chunks.*

**TL;DR:** For camera-conditioned world models, a reward must jointly grade *action execution* and *visual quality* over long videos. Judging a whole video against a full action sequence in one prompt dilutes short-lived local action evidence. WorldReward (Fudan / Tencent Hunyuan, 2026) splits paired videos into **action-aligned chunks**, has a VLM judge each chunk with structured evidence, then **votes** chunk decisions into video-level preferences on separate action and visual-quality axes.

**Prereqs:** [_rewards](_rewards.md), [cot-reward-model](cot-reward-model.md)
**Related:** [dpo](dpo.md), [grpo](grpo.md)

---

## What it is

A pairwise preference reward model targeted at long-horizon multimodal tasks (camera-conditioned world models specifically). Input: two candidate videos + their commanded action sequence. Output: a preference on each of two axes — action execution and visual quality.

## How it works

Three-stage pipeline:

1. **Chunk pairing.** Both videos are decomposed into short action-aligned chunks (one commanded action per chunk). Chunks align across the pair so decisions are made on comparable sub-problems.
2. **Structured evidence per chunk.** For each chunk, the VLM extracts structured visual evidence (before/after state, motion cues, artifacts) rather than free-form captions. This gives the judge a constrained input that resists the long-video dilution failure mode.
3. **Chunk-level pairwise judgment + voting.** The judge picks a preferred video per chunk on each axis; chunk decisions are aggregated by voting into video-level preferences separately for action and visual quality.

Training data for the reward model is a **reasoning-augmented preference dataset** built by (a) a frontier VLM generating structured judgments, (b) multi-turn tool-based agent auditing to catch inconsistencies, and (c) targeted human review of low-agreement cases. The reward model is trained to match these preferences.

## Why it matters

- **Aligns reward with what humans notice locally.** Short-lived action events (a hand touching the correct object for a beat) are exactly what dilutes in whole-video judging; per-chunk voting keeps them in the reward.
- **Enables RL post-training of world models.** Used for RL post-training of HY-WorldPlay 1.5, it consistently improves both action execution and visual quality across short and long horizons.
- **Generalizes beyond world models.** Any long-video generative task with a step-aligned action or event schema (video edits, driving simulators, embodied policies) can adopt the same chunk-and-vote wrapper.

## Gotchas & tricks

- **Chunk boundaries matter.** Poor chunking (misaligned across the pair, or too long) reintroduces the dilution problem; align on the commanded-action schema, not on uniform time slices.
- **Two axes, aggregated separately.** Do not collapse action-execution and visual-quality votes into one scalar — the axes disagree often and the training signal loses a dimension if merged.
- **Structured evidence beats free text.** The paper's ablation is: free-form chunk captions underperform the structured-evidence format.
- **Human review on low-agreement chunks only.** Full human labeling is prohibitive; concentrate labels where the automated audit disagrees.
- **Benchmark against strong general judges.** On WorldReward-Bench, WorldReward beats GPT-5.5 by +3.42 / +1.45 / +3.56 pp on action / appearance / motion — good sanity check that a specialized judge earns its cost.

## Sources

- Paper: *WorldReward: Reward Modeling for Camera-Conditioned World Models* — Wang et al., Fudan / Tencent Hunyuan / SJTU / SAIL, 2026 — https://arxiv.org/abs/2609.03952
- Benchmark: *WorldReward-Bench* — human-annotated reward-model agreement benchmark released with the paper.
