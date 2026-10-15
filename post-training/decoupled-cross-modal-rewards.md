# Decoupled Cross-Modal Rewards (AesCode)
*Depth — splitting a composite reward into independent correctness and aesthetic signals computed in different modalities.*

**TL;DR:** When training an LLM to generate code that renders into images (SVG, Manim, HTML/CSS, Three.js), a single scalar reward combining "code correctness" and "aesthetic of the rendered image" gets gamed: the policy over-weights whichever criterion is easier to improve. **AesCode** decouples the two: compute a code-side verifier reward and an image-side preference reward *separately*, then combine them at the gradient-aggregation step rather than inside the reward value. The policy learns both signals without one swallowing the other.

**Prereqs:** [_rewards.md](./_rewards.md), [grpo.md](./grpo.md), [rlvr.md](./rlvr.md)
**Related:** [rubric-rl.md](./rubric-rl.md) · [cot-reward-model.md](./cot-reward-model.md) · [../multimodal/README.md](../multimodal/README.md)

---

## What it is

A reward architecture for tasks where the generation is in one modality (code) but part of the quality criterion is in another (the rendered image). The two signals are kept *parallel* through the RL step rather than averaged into one scalar.

## How it works

**Two independent reward functions.**
- **Code-side reward `R_c`.** Verifier-style: does the code compile, does it render without error, does it pass syntactic / structural checks? A hard, RLVR-style signal.
- **Image-side reward `R_i`.** Preference-style: compare the rendered image against a reference (or score it with a learned aesthetic reward model). A soft, RLHF-flavored signal.

**Decoupled gradient aggregation.** Instead of `r = α · R_c + β · R_i` fed into one PPO-style update, compute two advantages `A_c` and `A_i` (each with its own group-normalization baseline), then combine their *gradients*:

```
∇θ L = w_c · ∇θ L(A_c) + w_i · ∇θ L(A_i)
```

The weights `w_c`, `w_i` are set once; because they live outside the reward value, they don't interact with the per-sample normalization that would otherwise let one reward dominate.

**Rollout structure.** For each prompt, sample `G` responses (code snippets). For each, render the image and compute `R_c` and `R_i`. Normalize within the group *per reward* — two separate group statistics, not one.

## Why it matters

- Breaks the "whichever reward is easier wins" pathology in GRPO-style multi-reward training. Decoupling makes the two signals orthogonal in the update, not just in nominal weighting.
- Reported: wins pairwise aesthetic preference against GPT-4-class visual-code generators without regressing on correctness.
- The template **generalizes well beyond aesthetic code**: any task where correctness and preference live in different output modalities (code → image, LaTeX → rendered math, SQL → table render) can plug in.

## Gotchas & tricks

- **Image reward model is the bottleneck.** `R_i` requires either a reference image (narrow use) or a learned aesthetic RM (which can itself be hacked). Keep the RM evaluation strict and audit outputs.
- **Rendering cost.** For every rollout you render; rendering can dominate step cost. Cache aggressively, use lightweight renderers in training and swap to full renderers at eval.
- **Weight schedule.** `w_c` and `w_i` can be annealed (correctness first, then aesthetic) to avoid the policy learning stylish but broken code.
- **Not a free pass to orthogonality.** The two gradients can still interfere at the parameter level; monitor the ratio of their norms.
- **Compare to naïve summing.** If `R_c` and `R_i` are already well-scaled and non-adversarial, naïve summing can be competitive; the decoupling pays off most when the two rewards have different reliability or different gradient magnitudes.

## Sources

- Paper: *AesCode: Aesthetic Code Generation with Decoupled Cross-Modal Rewards* — Gao, Yang, Li, Wang, Li, Tian, Cheng, Liu, Qiu, Dai, Chen, Dong, Tassiulas, Ying, Li, Luo — Microsoft / Yale / NTU / Pitt / SJTU / Fudan, 2026 — arXiv:2610.18860.
