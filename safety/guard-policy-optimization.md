# Guard Policy Optimization
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** An RL fine-tuning recipe for **moderation models** that guard computer-use agents (CUAs). Where prior moderation classifiers are trained by supervised classification on labeled action traces, Guard Policy Optimization treats the moderator as an *actor* whose reward comes from **intercepting real hazardous action sequences** in executable environments. Reported on HazardAuditor's benchmark suite as up to **+16.5 pp** safety-model accuracy over baselines.

**Prereqs:** [ppo](../post-training/ppo.md), [_rl](../post-training/_rl.md)
**Related:** [cot-monitoring](cot-monitoring.md), [action-trace-monitoring](action-trace-monitoring.md), [_attacks](_attacks.md)

---

## What it is

Moderator models for CUAs decide, per action step, whether to allow, block, or escalate an agent's proposed action (a shell command, a file write, a network call, a browser navigation). Supervised classifier moderators are trained on labeled traces of (action, hazard-label) pairs. This has two failure modes:

1. Labels are static, so the moderator cannot see how its own decisions change the downstream trace it will next be asked to moderate.
2. Cross-step hazards (a sequence of individually benign actions that is dangerous as a chain) are not directly rewarded.

Guard Policy Optimization frames moderation as an RL policy problem: the moderator's *policy* over allow/block/escalate is optimized in an execution-grounded environment where the reward penalizes hazardous outcomes observed after action rollout.

## How it works

The recipe (as described in the HazardAuditor framework) combines:

- **An execution-grounded environment.** Agent actions run in a real sandbox — a browser, a terminal, a filesystem, a set of external services — so downstream state is observable.
- **A normalized action representation.** Heterogeneous CUA interactions (a click, a shell command, a file write, an API call) are normalized into a common representation the moderator consumes.
- **A policy-optimization loop.** The moderator is treated as a PPO-style policy over allow/block/escalate; the reward is an aggregate of hazard signals observed after each decision (denied dangerous action → positive reward, allowed dangerous action → penalty, spurious block on benign action → penalty).
- **A dataset assembled from actual agent rollouts.** The training distribution reflects the CUA population in deployment, not a hand-curated benign/malicious split.

The result is a moderation model that is trained to make sequential decisions with an eye on rollout, not a per-step classifier.

## Why it matters

- **Sequential hazards are the actual threat model.** A cluster of individually-permissible actions (read config → decode env var → exfiltrate over HTTPS) is dangerous only as a sequence. RL over rollouts gives the moderator a shot at catching this; classification per action cannot.
- **Execution grounding closes the sim-to-real gap.** Rewarding on synthesized labels rewards labeling error. Rewarding on actual sandbox outcomes rewards catching real hazards.
- **+16.5 pp headline number.** On the HazardAuditor benchmarks, the RL recipe beats the strongest prior moderator baselines by up to 16.5 percentage points in safety-model accuracy — a large gap for a moderation task.

## Gotchas & tricks

- **Reward shaping is where this lives or dies.** Under-penalizing false blocks makes the moderator over-restrictive and the agent useless; under-penalizing false allows makes it unsafe. Reported ablations on this trade-off are the ones to read closely.
- **Sandbox fidelity is a rate limiter.** If the sandbox does not faithfully reflect the target deployment (e.g. no real credential handling, no realistic filesystem layout), the RL loop optimizes for the sandbox, not the world.
- **PPO-style updates can drift the moderator's language understanding.** Freezing lower LM layers, or using a KL-to-reference-model regularizer, keeps the moderator from over-specializing on the training distribution.
- **Coverage of the CUA population matters.** A moderator trained only on browser-agent rollouts will underperform on terminal-agent rollouts. The normalization across interaction types in HazardAuditor is specifically to avoid this.

## Sources

- Paper: *HazardAuditor: From Executable Threats to Safer Computer-Use Agents* — Feng, Lin, Wen, Guo, Ma, Wu, Deng, Ji — 2026 — [arXiv:2609.15134](https://arxiv.org/abs/2609.15134).
- Background: standard PPO — Schulman et al. — 2017 — the policy-optimization primitive Guard Policy Optimization specializes.
