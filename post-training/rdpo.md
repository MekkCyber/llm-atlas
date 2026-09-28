# Role-Decoupled Policy Optimization (RDPO)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A GRPO-family RL algorithm for **multi-role agent policies**. Each role (planner, synthesizer, executor, …) gets its own advantage estimate combining a **terminal outcome reward** with **turn-level rubric evaluations** on that role's turns only. Result: precise per-role credit assignment, avoiding the "one big advantage smeared over every token" failure mode of naive multi-role GRPO.

**Prereqs:** [grpo.md](grpo.md), [ppo.md](ppo.md), [_rewards.md](_rewards.md)
**Related:** [_rl.md](_rl.md), [rlvr.md](rlvr.md), [../agents/itersynth.md](../agents/itersynth.md)

---

## What it is

Modern deep-search agents (planner + synthesizer, planner + executor, planner + critic + …) run several distinct **roles** within one trajectory. If you apply GRPO to the concatenated trajectory, the group-relative advantage is a **scalar per rollout** broadcast across every token of every role — so a good planning step and a bad synthesis step both get credited with the trajectory's average. This is noisy and slow.

RDPO fixes credit assignment by computing role-specific advantages from role-specific reward signals.

## How it works

Given a trajectory with roles `R = {P, S, …}` and rollouts `i = 1..G` per prompt:

1. **Outcome reward** `r_out_i` — the standard terminal reward on rollout `i` (task success, verifier score, etc.).
2. **Turn-level rubric rewards** — for each role `r ∈ R`, score each of its turns against a role-specific rubric (planning correctness, synthesis integration quality, …), yielding `r_r_i,t`.
3. **Role advantage** — for each role, compute a group-relative baseline over its aggregated per-role reward: `A_r_i = (r_r_i − mean_j r_r_j) / std_j r_r_j` where `r_r_i` combines the outcome reward and this role's rubric rewards.
4. **Policy update** — apply the PPO-clipped update on each role's tokens using `A_r_i`, with a KL penalty to a reference model per role.

The rubric can be a rule-based scorer, a small judge LM, or a mix. Combining outcome + rubric per role is the paper's specific choice; other RDPO variants can weight the two differently.

## Why it matters

- Per-role credit turns "one noisy scalar per trajectory" into a set of cleaner, less-mixed learning signals; both convergence speed and asymptotic quality improve.
- Compatible with existing GRPO tooling: RDPO is the same clipped-ratio update, just with role-conditioned advantages.
- Applies wherever roles are distinguishable — planner/synthesizer, planner/executor, generator/critic, teacher/student.

## Gotchas & tricks

- **Rubric quality is the bottleneck.** A weak rubric collapses back into outcome-only RL. The paper uses turn-level rubrics that the base model can score reasonably.
- **Role imbalance.** If one role has many more turns than another, its advantage is over-smoothed. Normalize by turn count per role.
- **Role identification at training time.** The trajectory must expose which tokens belong to which role; format tags in the rollout make this cheap.
- **Group-relative baseline within each role.** Compute the mean/std over the *G* rollouts' per-role rewards, not over all tokens.

## Sources

- Paper: *IterSynth: Rethinking Deep Search Agents via Role-Decoupled Iterative Synthesis* — Wu, Yan, Lu, Chen, Zhang, Liu, Deng, Liu, Ma, Shao, Xiao, Shen — 2026 — [arXiv:2609.29444](https://arxiv.org/abs/2609.29444).
