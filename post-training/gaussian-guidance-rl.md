# Gaussian Guidance for Agentic RL
*Depth — sampling hint-prefix depth from a per-task Gaussian estimated online from rollouts you already have.*

**TL;DR:** Hint-based RL for long-horizon agents keeps a prefix of an expert trajectory before each rollout so exploration starts closer to success — but *how much* prefix to keep is task-dependent. Agent-G² (Wang et al., 2026) shows useful guidance occupies a **band** whose informativeness is approximately Gaussian around a per-task center, and samples the depth from that Gaussian using statistics estimated online from the same rollouts already used for policy optimization — no probe rollouts, no learned depth predictor.

**Prereqs:** [grpo.md](grpo.md), [_rl.md](_rl.md), [rl-prompt-curation.md](rl-prompt-curation.md)
**Related:** [rlvr.md](rlvr.md)

---

## What it is

Hint-based RL (a.k.a. teacher-forcing prefix injection) tackles reward sparsity in long-horizon agentic tasks by pre-filling the first *d* steps of each rollout with an expert trajectory, then letting the policy continue. Its effectiveness hinges on the **guidance depth** *d*: too shallow and the sparse-reward problem returns, too deep and the policy learns to imitate rather than improve.

Prior work treats *d* as a scalar. **Scheduled** methods share one *d* across all samples and ignore per-task heterogeneity. **Per-sample probing** estimates *d* per prompt with extra rollouts — accurate but expensive. Agent-G² proposes a third option: model *d* as a *distribution* per task, and estimate its parameters online from data you already have.

## How it works

The empirical finding, made from controlled sweeps: useful guidance depth doesn't concentrate at a single optimum, it fills a **band** whose informativeness profile is approximately Gaussian around a task-dependent center. So parameterize:

$$
d \sim \mathcal{N}(\mu_\text{task}, \sigma_\text{task}^2)
$$

with

- **Center** $\mu_\text{task}$ = global baseline + per-cluster difficulty (tasks are grouped by an unsupervised difficulty clustering, then the center adjusts around a global mean).
- **Spread** $\sigma_\text{task}$ tracks within-cluster reward variance — high variance ⇒ wider Gaussian ⇒ more exploration over guidance depths.

Both parameters are re-estimated online from the **rollouts already collected** for policy optimization (e.g. GRPO group rollouts). No separate probe rollouts, no learned depth predictor.

On Qwen2.5-1.5B / 7B-Instruct evaluated on ALFWorld and WebShop, Agent-G² beats the strongest hint-based, hint-free, and Aux-RL baselines on ALFWorld by 2.3 / 3.9 / 7.4 points at **under one-third the rollout cost** of per-sample probing.

## Why it matters

Curriculum-style hint RL has been stuck between two bad options: one scalar per run (blunt) or per-sample probing (expensive). Estimating the depth distribution from the rollouts you're already paying for is a cheap and correctly-shaped middle ground. Beyond hint depth, the Gaussian-band diagnostic — "useful settings for this hyperparameter form a task-dependent band, not a point" — is a reusable pattern for other RL scheduling hyperparameters (KL coefficient, entropy bonus).

## Gotchas & tricks

- **Requires group rollouts.** The online statistics come from a batch of rollouts per task; algorithms without grouping (single-rollout PPO) can't estimate the Gaussian parameters cheaply.
- **Per-cluster difficulty clustering matters.** The improvement over a single global Gaussian comes from adjusting the center per difficulty cluster; without it, you're back to a not-much-better global heuristic.
- **Gaussian is an approximation.** For tasks where the informativeness profile is genuinely bimodal (two disjoint useful depth regions), the Gaussian model under-serves both — the paper's experiments don't exhibit this but it's a foreseeable failure mode.

## Sources

- Paper: *Agent-G²: Gaussian Guidance for Agentic Reinforcement Learning* — Wang et al., 2026 — [arXiv:2608.23318](https://arxiv.org/abs/2608.23318)
