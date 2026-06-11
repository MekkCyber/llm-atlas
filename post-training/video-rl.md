# Video-RL

*Depth — RL post-training for video-language models with temporal-IoU, LLM-as-judge, and rule-verifiable rewards.*

**TL;DR:** Video understanding outputs aren't simple multiple-choice answers — they're temporal spans, dense captions, timestamps, counts. Each output type calls for a different reward design. Video-RL (as used in Keye-VL-2.0, 2026) combines temporal IoU for grounding, LLM-as-Judge for dense captioning, and rule-verifiable rewards on synthetic *FrameForge* videos where ground truth is constructed by hand. Together these cover the full video-understanding output space with reward signals that PPO/GRPO can train on.

**Prereqs:** [grpo](grpo.md), [rlvr](rlvr.md), [_rewards](_rewards.md)
**Related:** [mopd](mopd.md) · [reasoning/long-cot-rl](reasoning/long-cot-rl.md)

---

## What it is

Text RLVR has a clean reward story: math answer match, unit-test pass, exact string match. Video output types resist this:

- A *temporal grounding* output is a span $(t_\text{start}, t_\text{end})$ — naturally scored by IoU, not exact match.
- A *dense captioning* output is free-form text — needs an LLM judge.
- A *counting / timestamp / event reasoning* output needs ground-truth videos that humans haven't bothered to annotate yet.

Video-RL handles each with the appropriate reward design and runs them inside the same PPO/GRPO loop.

## How it works

### Temporal IoU for grounding

For temporal grounding tasks ("when does X happen in this video?"), the model emits $(\hat{t}_\text{start}, \hat{t}_\text{end})$ and the reward is the temporal IoU against the ground-truth span:

$$ r_\text{IoU} = \frac{|\hat{T} \cap T^*|}{|\hat{T} \cup T^*|} $$

Soft, dense, smoothly differentiable through the policy gradient. Doesn't have the "right or wrong" discontinuity that pure match-based rewards have.

### LLM-as-Judge for dense captioning

For dense-captioning tasks ("describe what happens in the video"), the model produces free-form text. Reward is a single scalar from an LLM judge that scores the caption against the video and (when available) a reference caption. Standard LLM-as-judge prompting; the judge is itself another LLM model held fixed during the RL run.

### Rule-verifiable synthetic videos (FrameForge)

For tasks where natural ground truth is scarce (precise timestamps, exact object counts, multi-step temporal reasoning), the paper introduces *FrameForge* — a synthetic video pipeline that constructs videos with hand-coded ground-truth properties (when each event occurs, how many objects appear, what the correct reasoning chain is). The reward is then a *rule-verifiable* function over the model's answer, exactly analogous to text RLVR for math.

### Composite reward in the RL loop

For each prompt, pick the appropriate reward family by task type. Mix in the standard GRPO/PPO update; the trust-region machinery is unchanged. Video-RL is therefore a *reward-design* contribution, not an algorithm contribution.

## Why it matters

- **Closes the video-RL gap.** Before, video post-training was mostly SFT — RL needed reward functions that didn't exist for most video tasks. Video-RL provides them.
- **FrameForge generalizes.** Any domain where natural ground truth is scarce can borrow the synthetic-with-rule-rewards pattern. Robotics videos, scientific simulations, agentic environments.
- **Mixes with text RL cleanly.** All three reward families produce a scalar that drops into GRPO; the video aspects don't require new algorithm work.

## Gotchas & tricks

- **Temporal IoU can be exploited.** Predicting a huge span that overlaps with many possible ground-truths exploits IoU. Pair with a length penalty or a strict-IoU floor.
- **LLM-as-Judge is noisy.** Judge LLMs prefer fluent-but-wrong captions over terse-but-correct ones. Sanity-check with a small human-labelled subset.
- **FrameForge correctness is your ground truth.** A bug in the synthetic pipeline becomes a baked-in wrong reward. Test the rule rewards on the synthetic data itself before RL.
- **Synthetic-to-real gap.** A model RL'd heavily on FrameForge may overfit to the synthetic style. Mix in real-video tasks with the other reward families.
- **Composite reward weighting is a hyperparameter.** When mixing IoU + judge + rule scores in one batch, normalize per family before combining or one family will dominate the gradient.

## Sources

- Paper: *Kwai Keye-VL-2.0 Technical Report* — Wen et al., Kwai/Kuaishou, 2026 — [arXiv 2606.10651](https://arxiv.org/abs/2606.10651).
- Related: temporal grounding evaluation lineage — TimeLens, Charades-STA.
