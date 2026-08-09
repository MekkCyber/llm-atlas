# VLM-as-Judge

*Depth — using a vision-language model as the reward/verifier for computer-use agent trajectories, and how it fails.*

**TL;DR:** Computer-use agent RL, evaluation, and data curation all rely on some function that decides whether a trajectory succeeded. Rule verifiers don't apply (the task is "did the agent complete the user's intent across screenshots?"); humans don't scale. The field's answer is VLM-as-judge: a vision-language model reads the screenshots + action log and emits a verdict. Empirically (OSReward, 2026) even frontier VLM judges share a **systematic leniency bias** that mislabels failed runs as successes, and the reliable ones cost too much to run at RL scale. Open trained reward models like OS-Shepherd (9B / 35B) close the affordability gap at ~30–60× lower cost than frontier judges.

**Prereqs:** [_rewards.md](_rewards.md), [rlvr.md](rlvr.md)
**Related:** [cot-reward-model.md](cot-reward-model.md), [reasoning/prm.md](reasoning/prm.md), [reasoning/orm.md](reasoning/orm.md)

---

## What it is

A specific instance of the general reward-model family: the "judge" takes a CUA trajectory (screenshots, DOM diffs, action log, user instruction) and returns a verdict. Compared to text-only judges, VLM judges are needed because the target signal — did the visible state of the machine reach the intended state — is often only readable from pixels.

Two closely related deployment shapes:

- **Evaluation.** Judge scores a batch of trajectories from candidate agents against ground truth; used to rank agents on benchmarks and to filter candidate models for release.
- **RL reward.** Judge scores rollouts during training; the score drives the policy update. This is where cost, latency, and reward-hacking risk all matter most.

---

## How it works

**Judge input.** Trajectory `τ = (instruction, [screenshot_1, action_1, ..., screenshot_T])`. Optional: a target checklist, a ground-truth final state, or a rubric.

**Judge output.** Usually a scalar in `[0,1]` (probability of success) or a categorical verdict (`success / partial / fail`). Optional: a written reasoning trace before the verdict — the CoT-RM style.

**Training data.**

- Trajectories from diverse CUA backbones executing human-verified instructions.
- Multi-stage human annotation to label whether each trajectory actually satisfied the instruction.
- Optional: reasoning traces from a stronger judge to teach the target judge how to justify its verdict.

**Cost profile.** A VLM judge call is O(N screenshots × per-image cost + per-token cost). For long trajectories this dominates the RL step budget; typical mitigations are downsampling frames, sending only the final state, or training a smaller open judge.

## Why it matters

- **Prerequisite for CUA RL at scale.** Without a cheap, trustworthy judge, verifiable-reward RL for computer-use agents just isn't affordable.
- **Reveals a shared failure mode.** OSReward finds *all* frontier VLM judges are lenient — they call failed runs successes. That's directly reward-hackable during RL: the policy learns to emit trajectories that look successful without actually succeeding.
- **Open reward models close the cost gap.** OS-Shepherd 9B/35B trained on OS-Shepherd-100K match commercial judges at 30–60× lower cost, changing what an academic-scale CUA RL run costs.

## Gotchas & tricks

- **Leniency bias is systematic.** Don't assume different frontier judges are independent errors; they aren't. An ensemble of frontier VLM judges will still miss failures at correlated rates. Include rule-based sanity checks (final-URL match, DOM snapshot equality) whenever possible.
- **Score with the full trajectory, not just the final screenshot.** Final-screenshot-only judging is cheaper and much more hackable — the policy learns to reach a "looks like success" state without doing the intermediate work.
- **CoT judges are more auditable.** A judge that first writes reasoning and then emits a verdict is inspectable when it fails — a debugging affordance the classic scalar-head judge doesn't offer. Similar tradeoff to [cot-reward-model.md](cot-reward-model.md) for math.
- **Rollout-time judging is where cost bites.** For RL, cheap open judges + rare frontier-judge audits is a common configuration.
- **Distribution shift.** A judge trained on one CUA backbone's trajectory style may over-approve rollouts from a very different backbone. Retrain or fine-tune when the training-time policy differs from the deployment policy.

## Sources

- Paper: *OSReward: Instituting Standardized Evaluation for Cross-Platform Computer-Use Reward Models* — Sun et al., HKU / OS-Copilot, 2026 — [arXiv:2607.28609](https://arxiv.org/abs/2607.28609). Introduces the OSReward benchmark, OS-Shepherd 9B / 35B, and OS-Shepherd-100K.
- Related: [cot-reward-model.md](cot-reward-model.md) covers the CoT-augmented reward-model pattern that VLM judges commonly adopt.
