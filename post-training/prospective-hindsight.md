# Prospective Hindsight (PH)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Before an agent sees the verifier's feedback on a rollout, it is asked to predict its own success. The RL update then up-weights rollouts where the predicted and actual outcomes disagree — penalizing both overconfident wrong answers and under-confident right answers. The policy becomes measurably better calibrated *while* improving task performance.

**Prereqs:** [rlvr](../post-training/rlvr.md), [grpo](../post-training/grpo.md)
**Related:** [_rl](../post-training/_rl.md), [_rewards](../post-training/_rewards.md), [ppo](../post-training/ppo.md)

---

## What it is

A calibration-aware reweighting that sits on top of any verifier-reward RL recipe (PPO, GRPO, RLVR). The policy emits a per-rollout success prediction before environment / verifier feedback; the training loss weights each rollout by the magnitude of the prediction-reality gap. "Prospective" because the forecast is before feedback; "hindsight" because the weight is applied after.

## How it works

1. **Rollout.** Sample a trajectory under the current policy.
2. **Self-prediction.** The policy also emits $\hat p \in [0,1]$: an estimate of success on this rollout. Implementation options: a dedicated head on the final hidden state, or a `<CONFIDENCE>` token prompted at trajectory end.
3. **Observe reward.** Get the verifier's binary (or scalar) outcome $r$.
4. **Compute gap.** $g = |r - \hat p|$. A confident wrong answer and an under-confident right one both score high.
5. **Reweight.** Scale that trajectory's policy-gradient contribution by $w(g)$, a monotone function (identity or a soft-max over a batch).
6. **Train $\hat p$** jointly with a BCE loss against $r$.

Compatible with GRPO's group-relative advantages: the group's reweighted-advantage mean is the baseline.

## Why it matters

Calibration is usually a post-hoc temperature trick that leaves the underlying model untouched. PH bakes calibration into the RL loop — surprising outcomes push the policy harder than confirmed ones, which is both the standard human-learning intuition and a direct way to concentrate compute where it matters.

Reported gains (vs the matching baseline RL method, several backbones):

- Consistent improvements on single-turn verifiable tasks.
- Consistent improvements on a multi-turn personal-agent benchmark.
- Measurably better calibration (lower ECE) at the same task performance.

The agent-side payoff: a well-calibrated agent can defer, re-plan, or ask for help where it is uncertain. Downstream orchestration systems (routers, humans-in-the-loop) can trust that confidence number.

## Gotchas & tricks

- Weighting by raw $|r - \hat p|$ is high-variance — the paper uses a group-normalized version.
- The self-prediction head is at risk of collapsing to a constant (always predict mean reward). A small entropy or variance bonus on $\hat p$ across a batch prevents this.
- Format matters: a `<CONFIDENCE>` token on the trajectory itself integrates cleanly with GRPO's token-level advantages; a separate head needs its own training signal.
- Needs a verifier (or at least a reliable scalar reward) — PH does nothing for RLHF-only setups where rewards are themselves noisy.

## Sources

- Paper: *Prospective Hindsight: Self-Calibrating Reinforcement Learning via Prediction–Reality Gaps* — Peng et al., Salesforce AI Research, 2026 — [arXiv:2610.02740](https://arxiv.org/abs/2610.02740)
