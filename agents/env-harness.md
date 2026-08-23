# EnvHarness
*Depth — a programmable plug-in layer that reshapes a static agent environment without touching its verifier.*

**TL;DR:** Agent-RL environments are usually hand-built and frozen — blind to the current policy's failure modes and stale as soon as the policy improves. EnvHarness wraps a static environment with **plug-in components at standard interfaces** so its behavior can be reshaped while the original verifier stays authoritative. A companion tool, **EnvRigger**, treats the target policy as a black box, watches its rollouts, diagnoses recurring flaws, and *synthesizes* new harness components that expose them — then validates on fresh rollouts. Across five benchmarks in four domains, EnvHarness beats both original environments and domain-specific env-generation pipelines by up to **+9.0 points** on held-out instances with 9.8% fewer execution steps.

**Prereqs:** [../post-training/_rl.md](../post-training/_rl.md), [../post-training/rlvr.md](../post-training/rlvr.md)
**Related:** [../post-training/rl-prompt-curation.md](../post-training/rl-prompt-curation.md), [../post-training/_post-training.md](../post-training/_post-training.md)

---

## What it is

Two layers on top of an existing agent environment:

- **Environment logic** (unchanged, verified) — the underlying task, action space, and reward-eligibility check.
- **Environment behavior** (mutable) — hooks into standard interfaces (observation shaping, action pre/post-processing, tool visibility, seed/state distribution) implemented as *harness components*.

Any reshaped environment must still route through the original verifier, so the RL reward remains trustworthy.

## How it works

1. **Diagnose.** EnvRigger runs the current policy on the static environment and inspects execution trajectories for repeated failure patterns (wrong tool, ignored constraint, hallucinated state, etc.).
2. **Synthesize.** For each diagnosed flaw, generate one or more harness components targeting it (e.g. a component that occludes a shortcut the policy was over-relying on, or one that injects a distractor observation exposing brittle grounding).
3. **Validate.** Roll out on the reshaped environment. Keep components that (a) still produce well-formed episodes, (b) shift the failure distribution as intended, and (c) preserve the original verifier's judgments on gold trajectories.
4. **Train.** Feed the reshaped rollouts to the RL loop. Because the verifier is unchanged, the reward channel is compatible with any RLVR-style trainer (PPO, GRPO).

## Why it matters

Agent RL is bottlenecked by the environments you can train against. EnvHarness turns environment construction into a **closed-loop, per-policy artifact** — much like reward-modeling turned "define the reward" into a learned artifact. Since a static verifier gates every harness component, the trick works without the reward-hacking risk of learned reward models.

## Gotchas & tricks

- The verifier must be well-covered by the reshaped state distribution — a harness component that pushes the policy into states the verifier scores incorrectly silently corrupts the reward.
- Diagnosis on a black-box policy is fragile if trajectories are short or the failure modes are heavy-tailed. The paper's five-benchmark evaluation is on relatively structured domains.
- Continuous co-evolution can drift: policy improves → old harness components no longer bite → EnvRigger needs to be re-run periodically, not once.

## Sources

- Paper: *EnvHarness: Awakening Static Worlds for Agent Learning* — Huang, Wang, Han et al., Google, 2026 — [arXiv:2608.19880](https://arxiv.org/abs/2608.19880)
