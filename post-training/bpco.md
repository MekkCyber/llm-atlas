# Best Practice Critic Optimization (BPCO)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** GRPO removes the value network (critic) by using the group mean as a baseline. But a reliable critic would let you get per-token advantages from a **single response** and — crucially — could be **conditioned on hidden information** the policy doesn't see (reference answers, grading rubrics). **BPCO** stacks five design choices — DPPO, bounded value predictions, Monte Carlo targets, unnormalized policy advantages, length-adaptive GAE — into a stable critic-based recipe that matches or beats GRPO while sampling one response per prompt. Introduced by Qi, Zhou, Lee 2026.

**Prereqs:** [ppo.md](ppo.md), [grpo.md](grpo.md)
**Related:** [_rl.md](_rl.md), [rlvr.md](rlvr.md), [_rewards.md](_rewards.md), [reasoning/long-cot-rl.md](reasoning/long-cot-rl.md)

---

## What it is

Critic-based PPO is unstable at LLM scale — the value network's early predictions are wildly off, and the policy update overshoots when advantages are computed from a bad value baseline. GRPO's response was to remove the critic and rely on group-mean baselines instead, at the cost of sampling $G$ responses per prompt.

BPCO isolates *why* critic-based RL is unstable and applies a targeted fix per failure mode. The result is a critic that's stable enough to replace group-relative estimation entirely.

## How it works

### The five design choices

1. **DPPO actor update.** Decoupled PPO — the actor loss and value loss don't share gradients through a common trunk, avoiding cross-contamination when either loss is early-unstable.
2. **Bounded value predictions.** The critic's output is clamped to the reward range. Prevents divergent value predictions from producing extreme advantages.
3. **Monte Carlo value targets.** Value targets are computed from actual sampled returns, not bootstrapped from the critic's own predictions. Slower to update but avoids the classical bootstrap-collapse failure mode of critic-based RL.
4. **Unnormalized policy advantages.** Skip the batch-wise advantage normalization that vanilla PPO uses. In LLM RLVR settings the advantage magnitudes carry signal about problem difficulty; normalizing washes it out.
5. **Length-adaptive GAE.** GAE's $\lambda$ implicitly assumes a fixed rollout length; long CoT rollouts break this. Length-adaptive GAE rescales the trace so bias-variance tradeoffs are consistent across variable-length responses.

### The critic can see what the policy can't

Because the critic exists only at training time, it can be **conditioned on hidden information**: the reference answer, a grading rubric, or teacher hints. The policy still sees only the prompt at inference. The critic uses the hidden info to give tighter value estimates than a policy-visible critic could.

For rubric-based rewards this is a real edge — the rubric is a rich supervision signal that can't be fed to the policy (it would leak into outputs) but is safe for a training-time-only critic.

## Why it matters

- **RL step-compute reduction.** BPCO uses one response per prompt where GRPO uses $G$. Even accounting for the critic's forward/backward, per-step compute drops meaningfully at long CoT lengths.
- **Rubric-conditioned critics.** The training-time-hidden-info trick applies immediately to rubric-based RL — the current standard for eval-driven capability training. Any signal you can compute at training time can now steer the policy through the critic.
- **Rehabilitates value-based RL for LLMs.** GRPO's dominance made "critic RL is unstable at scale" conventional wisdom. BPCO shows it was a stack of five fixable issues, not a fundamental barrier.

## Gotchas & tricks

- **MC targets need long rollouts to be stable.** Very short rollouts give the MC estimator few samples and high variance. For CoT ≥ 512 tokens BPCO is comfortable; for short-completion tasks bootstrap targets may still be needed.
- **Bounded value predictions collide with reward-scale drift.** If the reward range changes over training (curriculum, mixed reward types), the clamp becomes a live hyperparameter.
- **Length-adaptive GAE isn't a free lunch.** The rescaling has a hyperparameter of its own (how you scale $\lambda$ with length); the paper reports a schedule but not deep sensitivity.
- **Hidden-info critics can leak.** If the rubric is deterministic, a well-fit critic effectively encodes it in the advantage signal, which the policy can then absorb. For most rubrics this is desirable; for privacy-sensitive supervision, add noise to the critic input.

## Sources

- Paper: *Best Practice Critic Optimization* — Qi, Zhou, Lee, 2026 — introduces BPCO. [arXiv:2608.23566](https://arxiv.org/abs/2608.23566). Code: [github.com/QPHutu/golden_critic](https://github.com/QPHutu/golden_critic).
- Related: *Proximal Policy Optimization* — Schulman et al., 2017 — the PPO baseline BPCO stabilizes.
- Related: *DeepSeekMath* (GRPO) — Shao et al., 2024 — the group-relative alternative BPCO matches with one response per prompt.
