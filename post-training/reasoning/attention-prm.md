# Attention Process Reward Model (APRM)
*Depth — process rewards computed from hidden states and attention maps instead of textual steps.*

**TL;DR:** Standard PRMs score each reasoning step from its text, which suffers from **textual reward aliasing**: the same words get the same reward regardless of the underlying computation. APRM (2026) instead reads the policy's own hidden states and attention maps at each step and produces a scalar. With only ~46M parameters, trained with ranking + counterfactual objectives, it gives an average pass@8 gain of 2.88% on top of representative RL methods.

**Prereqs:** [prm](prm.md), [_rewards](../_rewards.md)
**Related:** [cot-reward-model](../cot-reward-model.md), [long-cot-rl](long-cot-rl.md), [rlvr](../rlvr.md)

---

## What it is

A PRM variant whose input is not the token sequence of a reasoning step but the policy's **internal signals**: the layer-wise hidden states at step boundaries and the attention maps leading into the step's output tokens. A small transformer head (~46M params) converts these into a scalar step reward.

## How it works

Input featurization per step:

- Pool hidden states across the step's token span, at selected layers.
- Flatten / attention-pool the attention maps that produced those tokens.
- Concatenate to form the APRM input.

Training objectives:

1. **Ranking.** Given pairs of step sequences (correct vs. corrupted trace from the same prefix), rank correct higher.
2. **Counterfactual.** For a step with a known good continuation, train the reward to drop when key hidden states are patched to a corrupted version — enforcing that the signal depends on *internal* reasoning features, not just surface correlates.

Integration with RL: APRM's per-step reward is combined with the outcome reward (RLVR-style) and consumed by GRPO/PPO. APRM is refreshed periodically from the current policy to handle distribution shift.

## Why it matters

Direct attack on **reward hacking of textual PRMs**. Any PRM that reads only the text can be fooled by surface rewrites that preserve wording but change computation (or vice versa). Hidden-state rewards are harder to game with surface edits because they reflect what the model actually computed. The 2.88% pass@8 lift is small in absolute terms but comes with a tiny head and no extra text labels — a compelling cost/benefit if it holds at scale.

APRM also connects to safety monitoring: the same internal signal is a probe into reasoning quality, which could be repurposed for detection as well as training.

## Gotchas & tricks

- **Drift sensitivity.** Hidden-state distributions move faster than text during RL. APRM must be refreshed on-policy; a stale APRM degrades quickly.
- **Policy coupling.** The reward reads the policy's internals, so an APRM trained for one model does not transfer to another. Each model family needs its own APRM.
- **Architectural constraints.** Pooling scheme and layer selection matter; the paper's ablations show late-layer states dominate.
- Interpretability of the reward is weaker than for text PRMs — you can't easily look at a step and say "APRM rewarded it because X."

## Sources

- Paper: From Rewarding What Models Say to How They Think: Attention Process Reward Model for LLM Reinforcement Learning — Han et al., 2026 — [arXiv:2611.30627](https://arxiv.org/abs/2611.30627)
