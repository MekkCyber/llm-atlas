# Faithful-RFT
*Depth — verifiable-reward RFT stage from TLive-Omni for omni-modal understanding.*

**TL;DR:** Faithful-RFT is the reinforcement fine-tuning stage in TLive-Omni. Instead of optimizing for exploration during rollout (as in reasoning-style RL), it scores the **final answer** directly with task-verifiable feedback. Pairs this with a *dynamic sampling* trick that regenerates rollout groups whose reward variance collapses to near zero — a common GRPO failure mode.

**Prereqs:** [grpo](grpo.md), [rlvr](rlvr.md)
**Related:** [_rl](_rl.md), [rejection-sampling](rejection-sampling.md)

---

## What it is

A post-SFT RFT stage for omni-modal LLMs where the reward is the *end answer's task-verifiable score* (correct product ID, correct temporal grounding, faithful transcription) — not a reasoning-progress signal on intermediate steps. Designed for streams (video + audio + text + images) where correctness is checkable but reasoning is not.

## How it works

- Rollouts are generated for each prompt group as in GRPO.
- The **only reward** is a task-verifiable score on the final response (grounding accuracy, faithful QA, etc.). Intermediate tokens get no shaping.
- **Dynamic sampling for zero-variance groups:** if all rollouts in a group receive the same reward, the group carries no relative-advantage signal. Faithful-RFT detects this and *regenerates* the group with fresh samples (typically at a higher temperature) so the advantage estimator stays useful.
- Runs on top of a supervised recipe that already teaches instruction-following, so exploration outside the desired answer format is suppressed by the SFT prior.

## Why it matters

For multimodal understanding, task-verifiable feedback is available (there's a ground-truth product, timestamp, or transcription) but reasoning-style RL over-explores and hurts faithfulness. Faithful-RFT keeps the RL signal cleanly aligned with what you want to measure at deploy time, and the dynamic-sampling trick is broadly useful anywhere GRPO's group-advantage collapses.

## Gotchas & tricks

- Zero-variance regeneration has a compute-budget cap in practice — after N regenerations, accept the group and move on. Otherwise a stuck prompt can starve the trainer.
- Works best when the verifier is cheap (regex, exact match, IoU) — if the verifier is another LLM, cost dominates.
- Not a replacement for reasoning-RL; complementary for answer-faithfulness in tasks where reasoning traces aren't the eval target.

## Sources

- Paper: *TLive-Omni: An Omni-Modal Understanding Model for E-Commerce Live Streaming* — Hu et al., 2026 — [arXiv:2608.20958](https://arxiv.org/abs/2608.20958)
