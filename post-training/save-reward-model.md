# SAVE — Self-supervised RM Improvement via Value-Anchored On-policy Feedback

*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A method for *co-training* the reward model with the policy during RLHF. The value head (already trained as part of the RL setup) grades the policy's on-policy responses; a prompt-specific value head acts as an *adaptive anchor* for reward scale; RM advantages filter out ambiguous samples; the RM is updated via a contrastive objective on the rest. The point is to keep the RM aligned with the evolving policy without collecting new human preferences — fighting RM staleness that grows as the policy drifts past the static RM training distribution.

**Prereqs:** [_rewards](_rewards.md), [_rl](_rl.md), [ppo](ppo.md), [grpo](grpo.md)
**Related:** [dpo](dpo.md) · [cot-reward-model](cot-reward-model.md)

---

## What it is

Classical RLHF freezes the reward model at the start of RL. The RM was trained on preferences over responses from an *earlier* policy (often the SFT checkpoint). As RL proceeds, the current policy generates responses that look less and less like RM-training distribution, and the RM's judgments on those out-of-distribution responses become unreliable. The standard cure — collect new preferences from humans, retrain the RM, repeat — is expensive and slow.

SAVE proposes an alternative: use the *value head* (the per-state value estimate, $V(s)$, that PPO-family algorithms train anyway) as a self-supervised signal to improve the RM on the policy's current distribution. The value head is already learning to predict expected reward-to-go from on-policy responses, so it carries information about which current-policy responses are good vs bad — information the static RM was never trained on.

---

## How it works

### Pipeline

```
1. RL loop emits batch of on-policy responses {o_i} for prompts {q_i}.
2. Value head V(q, o)  outputs a scalar value estimate per response.
3. Prompt-specific value head v(q) outputs an anchor per prompt
   (a calibration term that adapts the scale of v across prompts).
4. RM advantage  A_RM(q, o)  =  V(q, o)  −  v(q)
5. Filter samples with |A_RM| below a threshold (ambiguous / uninformative).
6. For the remaining responses, derive a preference signal:
       within each prompt, A_RM > 0 ⇒ "good", A_RM < 0 ⇒ "bad".
7. Update RM via contrastive loss (Bradley-Terry-style) on these
   on-policy pseudo-preference pairs.
```

The RM is now being updated on the *current* policy's response distribution, with labels derived from the value head — no new human annotation in the loop.

### Why the value-head signal is meaningful

The value head is trained by TD-learning against the actual reward — so it is, in a literal sense, *the policy's own estimate of which responses are likely to score well under the current RM*. As the policy drifts, the value head tracks. Using V(q, o) - v(q) as a self-graded label is closer to "what would the RM say if it could see this response" than to a random pseudo-label.

### The prompt-specific anchor

A flat threshold on V(q, o) doesn't work because different prompts have wildly different reward scales (hard prompts score lower across the board; easy ones higher). The prompt-specific value head v(q) normalizes this. The advantage A_RM = V(q, o) - v(q) is comparable across prompts, which makes the filter and the contrastive labels well-calibrated.

### Filtering ambiguous samples

Samples with small |A_RM| are the value head's "don't know" cases — supervising the RM on them is noise. Dropping them sharpens the contrastive signal.

---

## Why it matters

- **Addresses RM drift without new human data.** This is the practical RLHF failure mode at scale: the longer you train, the more your RM mis-grades your policy. SAVE plugs the hole.
- **Plug-and-play across RL algorithms.** Validated under GRPO, RLOO, and GSPO on multiple backbones — wherever there's an estimate of state value, SAVE applies.
- **No new infrastructure.** Reuses the value head you already train. The added cost is a small RM update per RL step.

---

## Gotchas & tricks

- **Requires a competent value head.** If V is poorly trained early in RL, its self-supervision is garbage and the RM degrades. Warm up V before turning SAVE on.
- **Prompt-specific anchor is critical.** Without it, you're contrasting absolute value scores across heterogeneous prompts, which is mostly noise.
- **The ambiguity-filter threshold is a tradeoff.** Too aggressive (filter only big-margin examples) gives sparse RM updates. Too lax (keep small margins) corrupts the RM with low-confidence labels.
- **Doesn't solve reward hacking.** SAVE keeps the RM aligned with the value head, which is aligned with the RM's own reward signal. If the RM is hackable, SAVE will track that. Pair with safety/auxiliary signals; don't rely on SAVE as a hack-defense.
- **Not a substitute for periodic real-human preference refresh.** SAVE buys you longer training horizons between human-preference collections — it doesn't replace them.

---

## Sources

- Paper: *The Flip Side of RLHF: On-Policy Feedback for Reward Model Self-Supervised Improvement* — Wang, Wu, Tang, Li, Liu, Zheng, 2026 — USTC / BIGAI. Introduces SAVE; validates across GRPO/RLOO/GSPO on six benchmarks and multiple policy backbones.
