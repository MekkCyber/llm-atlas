# Rubric Rewards
*Depth — instance-aware multi-axis judge rewards for RL on open-ended generation.*

**TL;DR:** For open-ended generation tasks with no ground truth (SVG code, UI layouts, creative writing), scalar rewards (CLIP, aesthetic scorers) transfer poorly and get reward-hacked. Rubric rewards convert each *instruction* into a small **instance-specific rubric** — a handful of scorable axes derived from the prompt itself — and use a VLM/LLM judge to score each rollout item-by-item against that rubric. The weighted per-item satisfaction score is the reward, typically consumed by GRPO. Introduced at scale by RULER (2026) for SVG generation.

**Prereqs:** [grpo.md](./grpo.md), [_rewards.md](./_rewards.md)
**Related:** [rlvr.md](./rlvr.md), [cot-reward-model.md](./cot-reward-model.md), [_rl.md](./_rl.md)

---

## What it is

A reward-design pattern for RL post-training on tasks where (1) there is no reference output to check against and (2) scalar aggregate metrics are known to invite reward hacking. The rubric is *derived from the instruction* — no paired ground-truth outputs, no human preference labels — and the judge scores rollouts one rubric item at a time. It sits alongside verifiable rewards ([RLVR](./rlvr.md)) and CoT reward models ([cot-reward-model.md](./cot-reward-model.md)) as a third reward family, aimed at open-ended tasks with no verifier.

## How it works

Three-stage pipeline per training example:

1. **Rubric extraction.** Prompt a strong LLM/VLM with the instruction; ask it to enumerate a fixed-size rubric (RULER uses 6 items spanning semantic, visual and stylistic axes). Each item is a scorable question grounded in the instruction — e.g., "Does the image contain a red circle?" or "Are the stylistic descriptors respected?"
2. **Item-by-item scoring.** For each rollout, ask a judge model to answer each rubric item on a 0–1 (or bounded) scale, reading the rendered/decoded artifact. Weighted sum → per-rollout scalar reward.
3. **RL.** Plug the reward into standard GRPO. The rubric changes *per prompt*; the axis weights are fixed hyperparameters.

Because the rubric is derived from prompt text, this needs no paired ground truth and no human preferences. Ablations in RULER pin **rubric design** (which axes, how they're phrased) as the active lever — swapping the judge for a scalar CLIP score collapses back into reward hacking.

## Why it matters

- **RL on domains without ground truth.** Creative writing, UI code, SVG, layout, prose. Any task where "did it succeed" is a multi-axis judgment call.
- **Reward hacking resistance.** Multi-axis rubric scoring decomposes reward; the policy can't collapse into a single feature that maximizes a scalar (a common failure of CLIP-as-reward).
- **Cheap to author.** Rubrics are text; new domains only need a good prompt for rubric extraction and a capable judge.

## Gotchas & tricks

- **Judge distribution shift.** Item-by-item scoring is much easier for judges than free-form preference scoring, but bias in axis wording still propagates. Sanity-check by rewording axes and checking rank stability.
- **Rubric size.** Too few axes → back to scalar-reward pathology. Too many → per-rollout scoring cost dominates.
- **Judge cost.** The judge is called $G$ × (rubric size) times per prompt (with $G$ the GRPO group size). This is the largest cost item; smaller judges + cascading (see [judge-cascade.md](../evaluation/judge-cascade.md)) helps.
- **Instance-derived ≠ instance-perfect.** The judge that extracts the rubric can miss axes that matter for a particular prompt. Cross-checking with human preference on a held-out slice is worth doing at least once per domain.

## Sources

- Paper: *RULER: Instance-aware Rubric Rewards for SVG Generation* — Ran, Zheng, Zhang, Lin, Peng, 2026 — [arXiv:2609.25270](https://arxiv.org/abs/2609.25270).
- Related: *DeepSeekMath* — Shao et al., 2024 — GRPO, the RL algorithm rubric rewards typically feed.
