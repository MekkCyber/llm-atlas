# On-Policy Distillation (OPD)

*Depth — teacher-student distillation where the student's own multi-turn rollouts drive supervision, with Replayed-Prefix OPD as the compute-efficient variant.*

**TL;DR:** In agentic settings a student LLM interacts with an environment across many turns; on-policy distillation lets a teacher provide dense per-step supervision *along the student's own trajectory* rather than on curated teacher rollouts. **Replayed-Prefix OPD (ReOPD)** removes the per-update environment cost by pre-collecting teacher trajectories and letting the student act at selected steps within replayed prefixes — the teacher then labels those steps without new environment interactions.

**Prereqs:** [_rl.md](./_rl.md), [_post-training.md](./_post-training.md)
**Related:** [rejection-sampling.md](./rejection-sampling.md) · [grpo.md](./grpo.md) · [../systems/partial-rollouts.md](../systems/partial-rollouts.md)

---

## What it is

Standard imitation learning trains a student on teacher trajectories — safe, but the student never encounters the states it will actually visit at deployment. Fully online OPD fixes the distribution mismatch by rolling out the *student* through the environment and querying the *teacher* at every visited history. This works but is expensive: each gradient step needs fresh student rollouts and fresh teacher queries.

ReOPD is the middle position: reuse a fixed pool of teacher trajectories as **replayed prefixes**; at each prefix step, ask the student to act; ask the teacher to label the student's step; step-decay how often you take the student's action deeper into the prefix.

## How it works

Given a teacher trajectory $\tau = (s_0, a_0^T, s_1, a_1^T, \ldots)$ pre-collected once:

1. Sample a step $k$ from a **step-decaying schedule** $p(k) \propto \gamma^k$ that prefers early positions.
2. Roll the prefix $(s_0, a_0^T, \ldots, s_k)$ deterministically from the log.
3. Let the student act: $a_k^S \sim \pi_\text{student}(\cdot \mid \text{prefix})$.
4. Query the teacher at that mixed history for the target distribution $\pi_\text{teacher}(\cdot \mid \text{prefix} \cup a_k^S)$ and take an SFT/KL step.

No fresh environment execution is required — the prefix bytes come from disk, the student's action is scored by the teacher without being carried forward into an actual next state.

## Why it matters

**On-policy signal without on-policy rollouts.** The student is graded on its own actions, but the surrounding trajectory is the teacher's — an approximation that is exact for the current step and biased for the future.

**Names the prefix trap.** As you push $k$ later in the trajectory (more student-on-policy), the mixed history drifts away from the teacher's training distribution and the teacher's target becomes unreliable. Sampling every step uniformly maximises *relevance to the student*; sampling only step 0 maximises *reliability of the teacher* — the step-decaying schedule is an explicit knob on that tradeoff.

**Cheaper than full OPD.** Removes the per-update environment cost, which for real multi-turn agentic tasks (SWE tools, computer use, shell) dominates training wall-clock.

## Gotchas & tricks

- **Schedule choice matters more than $\gamma$ value.** Any monotonically decreasing $p(k)$ that leaves non-zero mass at step 0 recovers most of the reliability gain; the exact functional form is a second-order knob.
- **Teacher reliability is per-step, not per-trajectory.** Even at step 0 the teacher may be uncertain (open-ended reflexion turns). Use teacher log-prob entropy as a per-step confidence filter if available.
- **Off-environment ≠ off-policy-safe.** ReOPD still commits the classic imitation-learning error: rare student-only recovery moves are never rehearsed because the prefix always came from the teacher.
- **Composes with rejection sampling.** Filter teacher trajectories by outcome success before using them as prefixes — a bad teacher trajectory is worse than a short one.

## Sources

- Paper: *Multi-Turn On-Policy Distillation with Prefix Replay* — Liao, Dong, Monz, Xu, Dong, Wei — Microsoft Research / University of Amsterdam, 2026 — introduces ReOPD and the prefix-trap framing.
- Related: on-policy vs off-policy tradeoffs in [ppo.md](./ppo.md) and [grpo.md](./grpo.md).
