# Self-Evolving Reward Calibration
*Depth — automatic re-weighting of a multi-component reward across training iterations.*

**TL;DR:** RLHF on multi-component rewards (e.g. visual quality + motion + text-alignment + emotion for video generation) typically fixes the mixture weights by hand. **EmoReward's calibration loop** replaces the manual grid search: after each RL training iteration, it estimates how *reliable* each reward signal was on that iteration (via instance-level gating and cross-signal agreement) and adjusts the mixture weights accordingly for the next iteration. The reward *models* are frozen; only their *weights* evolve.

**Prereqs:** [_rl.md](./_rl.md), [grpo.md](./grpo.md), [_rewards.md](./_rewards.md)
**Related:** [cot-reward-model.md](./cot-reward-model.md) · [rl-prompt-curation.md](./rl-prompt-curation.md)

---

## What it is

Multi-reward RLHF uses a weighted sum:

$$
R(x) \;=\; \sum_i w_i \, r_i(x)
$$

The $r_i$ are specialized reward models (one per axis). The $w_i$ are picked once, usually by practitioners sweeping manually on a small eval set. This is fragile: the right weights depend on which rewards are currently informative for the training distribution, which drifts as the policy learns.

Self-evolving reward calibration treats the $w_i$ as **learned parameters of the training loop itself**, updated between iterations from measurements on the previous iteration's rollouts.

## How it works

**Instance-level gating.** Each $r_i$ comes with a confidence predicate — e.g. "trust face-emotion reward only on frames with face-detection confidence > τ." Rollouts where a signal's predicate fails are masked out *for that signal*, not for others.

**Reliability estimation.** After an iteration, measure each $r_i$'s *reliability* on the rollout buffer. In EmoReward this is a composite of:

- agreement with other active signals (cross-reward correlation);
- within-signal variance across rollouts (low-variance signals that are "always 1" aren't guiding anything);
- an ablation delta (would removing $r_i$ have changed the top-ranked rollouts?).

**Weight update.** The $w_i$ are rescaled from the reliability estimates — for example, $w_i \leftarrow \text{softmax}(\log w_i + \eta \, \hat{\rho}_i)$ with $\hat{\rho}_i$ the reliability score and $\eta$ a step size. This happens **between** RL iterations, not inside them.

**No reward model retraining.** The individual $r_i$ are frozen throughout. Only the mixture coefficients change.

## Why it matters

Reward-mix tuning is one of RLHF's most under-documented pain points. Every production RLHF system does it; nobody writes about how. A principled, data-driven weight loop (a) removes a lot of manual sweeping, (b) adapts automatically when the policy outgrows one of the reward signals, and (c) is reward-model-agnostic — it works on top of whatever $r_i$ you have. The technique is not specific to emotional video and plugs into any GRPO/PPO-style loop with ≥ 2 reward components.

## Gotchas & tricks

- Reliability estimation needs a nontrivial eval buffer per iteration; too few rollouts and the weight update is noise.
- If one reward is both reliable *and* easy to game, the loop will up-weight it into a reward-hacking spiral. Keep a held-out "diversity" eval as a stopgap.
- Soft-reset the weights periodically — compounding small updates over many iterations can drift into degenerate mixtures.

## Sources

- Paper: *EmoReward: Self-Evolving Reward Calibration for Emotional Video Generation RL* — Zheng et al., 2026 — https://arxiv.org/abs/2611.06518
