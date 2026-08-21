# VA-Judger
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A **chain-of-thought reward model** for joint video+audio generation, trained from human preference feedback over paired (prompt, video, audio) triples. Existing rewards combine per-modality metrics (video FVD, audio FAD, sync scores) and miss the *joint* semantic/temporal coherence that shapes human judgement. Huang et al. (2026) contribute a preference dataset, a benchmark, and the CoT reward model itself — a coherent single-scalar reward for RL post-training of A/V generators.

**Prereqs:** [_rewards.md](./_rewards.md), [cot-reward-model.md](./cot-reward-model.md)
**Related:** [../multimodal/README.md](../multimodal/README.md), [dpo.md](./dpo.md)

---

## What it is

A **joint** cross-modal reward model. Inputs: (text prompt, generated video, generated audio). Output: a scalar preference score aligned with human raters over the same triples, produced through an explicit chain-of-thought critique.

The framing rejects the standard "reward = weighted sum of per-modality metrics" approach because such sums are blind to the semantic/temporal relationships *between* modalities — e.g. audio that matches the prompt but leads the video by 300ms, or perfect visuals with plausible-but-wrong ambient audio.

## How it works

1. **Preference dataset.** Human raters compare pairs of A/V generations for the same prompt and pick a preferred triple. Captures cross-modal judgement (coherence of the joint output), not per-modality quality alone.
2. **CoT reward model.** A multimodal LLM that ingests (prompt, video tokens, audio tokens) and generates a short reasoning trace before emitting the scalar preference score. The trace itself is supervised by human-written justifications on a subset of pairs.
3. **Benchmark.** A held-out evaluation set with human-labelled preferences and a compatible protocol for scoring reward models.
4. **Downstream use.** The reward model plugs into RL post-training loops (PPO/GRPO/DPO-style) as the single reward signal for joint A/V generators.

## Why it matters

- **Unlocks RL post-training for joint A/V generation.** The bottleneck for RLHF on A/V models has been the reward: composite metric sums are noisy proxies for human joint preference. A learnt joint reward is what makes RL feasible here.
- **CoT trace is auditable.** Emitting reasoning before the score gives a signal for debugging reward-hacking and for downstream RLAIF (reward-model critiques as training data).
- **Sets a benchmark others can compete against.** The paired dataset + protocol lets future joint reward models compare on the same axis.

## Gotchas & tricks

- **CoT length vs cost tradeoff.** Longer traces improve reward-alignment but multiply per-sample cost during RL rollouts. Cap the trace budget for training loops.
- **Preference collection bias.** Raters familiar with A/V media may over-weight sync artefacts; raters recruited from general-purpose crowd platforms may under-weight them. Both bias the reward model.
- **Composition with per-modality metrics.** VA-Judger *replaces* the joint-coherence term, not the per-modality quality checks. Keep audio/video quality regularisers if you need them.
- **Reward hacking mode: hallucinated grounding.** Models can learn to generate visually spectacular scenes with generic prompt-conforming captions — VA-Judger's CoT should be inspected on rollouts to catch this early.
- **Dataset generation to English/short-prompt only in the paper.** Multilingual or long-narrative A/V may need additional preference data.

## Sources

- Paper: *VA-Judger: Reward Modeling from Human Preference Feedback for Joint Video-Audio Generation* — Huang et al., Fudan / Huawei Noah's Ark, 2026 — [arXiv 2608.18607](https://arxiv.org/abs/2608.18607) — introduces the dataset, benchmark, and CoT reward model.
