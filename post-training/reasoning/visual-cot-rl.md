# Visual-CoT RL
*Depth — RL post-training on reasoning chains that live in pixel space rather than in text.*

**TL;DR:** Train a multimodal model to reason inside a visual chain-of-thought — a sequence of imagined frames — using group-relative RL with paired global outcome rewards and per-step local rewards for physical and logical consistency. No paired image-text supervision. Anchored by VR-GRPO (UniVR, 2026) on the VR-X benchmark for cross-source visual reasoning.

**Prereqs:** [../grpo](../grpo.md), [prm](prm.md)
**Related:** [long-cot-rl](long-cot-rl.md), [../rlvr](../rlvr.md), [../../multimodal/README](../../multimodal/README.md)

---

## What it is

Standard reasoning RL rewards a model for producing a correct *textual* chain of thought that terminates in a correct answer. Visual-CoT RL applies the same idea to a *visual* chain: the model produces intermediate imagined frames (or visual latents) that constitute the reasoning trace, and the RL objective scores them for logical coherence, physical plausibility, and terminal task success.

The point is to train visual reasoning without paired image-text supervision — a live open question, because image-text pairs are the usual crutch for multimodal training.

## How it works

### VR-GRPO

VR-GRPO (UniVR, 2026) is the current anchor. It keeps GRPO's group-relative baseline (compare the return of each rollout in a group to the group mean, no separate value network) and scores each rollout at two granularities:

1. **Global reward.** Task success at the end of the visual reasoning chain (e.g. did the imagined trajectory reach the goal state, did the puzzle get solved).
2. **Step-level reward.** Per-step checks on the imagined frames — physical plausibility (does the imagined next frame respect object permanence, gravity, contact), logical coherence (does the imagined transition follow the intended sub-goal).

The step-level reward is what makes the recipe distinct: it densifies the RL signal without any language-space supervision.

### Purely visual protocol

Training and evaluation avoid image-text pairs. The reward comes from (a) task success on physical tasks (which is measurable in pixel space) and (b) automated consistency checks on the imagined frames. This is what unlocks the "learn from pure visual demonstrations" claim.

## Why it matters

- **Removes the paired-supervision requirement.** Image-text pairs are expensive and biased toward what humans caption. Visual-CoT RL learns from tasks with measurable pixel-space outcomes.
- **Generalizes group-relative RL past text.** GRPO's group-relative baseline was designed for text CoT but transfers cleanly to visual rollouts — one of the first clean demonstrations that group-relative advantages are modality-agnostic.
- **Improves standard multimodal benchmarks as a side effect.** UniVR reports gains on unrelated multimodal understanding tasks, suggesting visual-CoT training induces a more usable visual reasoning backbone.

## Gotchas & tricks

- **Step-reward quality is the whole game.** A weak physical-consistency check silently rewards degenerate visual chains. The reward function is the model — spending training-compute on the reward, not just the policy, matters.
- **Frame budget.** Every step in the visual CoT is a rendered frame — much more expensive per token than text. Chain lengths are typically short (5–20 frames), pushing the reward function to catch violations early.
- **Cross-source data.** VR-X pulls from 16 visual sources; single-source visual reasoning benchmarks under-report the difficulty. Evaluate on cross-source suites to avoid overfitting to one rendering style.
- **Not a substitute for text CoT.** Visual-CoT RL wins on tasks that live in pixel space (manipulation, spatial puzzles, physical reasoning). Language-heavy reasoning still benefits from text CoT.

## Sources

- Paper: *Thinking in Visual Space for Unified Visual Reasoning* — Wei et al., 2026 (BJTU / ByteDance) — VR-GRPO and the VR-X benchmark.
- Related: [../grpo.md](../grpo.md), [prm.md](prm.md), [long-cot-rl.md](long-cot-rl.md).
