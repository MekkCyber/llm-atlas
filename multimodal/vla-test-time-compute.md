# VLA Test-Time Compute
*Depth — spend inference compute at the high-level planner of a hierarchical VLA by rolling out candidate subtasks through a learned world model.*

**TL;DR:** Hierarchical vision-language-action (VLA) models typically pick each subtask with a single planner forward pass, leaving no way to spend more compute on hard branches. VLA test-time compute reframes planning as *world-model-guided search*: sample N candidate subtasks, roll each forward in an action-conditioned video world model, score the imagined trajectories against the language goal, and execute the best. Same o1-style "think longer on hard problems" idea, applied to embodied planning. τ₀-VLA reports monotonic scaling with rollout budget on long-horizon manipulation.

**Prereqs:** [README.md](README.md), [../post-training/reasoning/mcts.md](../post-training/reasoning/mcts.md)
**Related:** [../post-training/reasoning/long-cot-rl.md](../post-training/reasoning/long-cot-rl.md), [_vla.md](_vla.md)

---

## What it is

Two distinct compute axes for a hierarchical VLA:

- **Low-level (action head)** — decode the next action given the current subtask. Latency-bound; compute here has hard limits.
- **High-level (planner)** — pick the next subtask given the goal, current observation, and history. Traditionally one forward pass; conceptually deliberative.

VLA test-time compute keeps the low-level head fast and pushes the deliberation up into the planner. The planner spends variable compute per decision, proportional to how consequential the branch is (goal-completion, phase transitions, high-cost mistakes).

## How it works

```
subtasks_candidates = planner.sample_k(state, goal, k=N)
scores = []
for c in subtasks_candidates:
    imagined_traj = world_model.rollout(state, subtask=c, horizon=H)
    scores.append(goal_match(imagined_traj, goal))
best = subtasks_candidates[argmax(scores)]
execute(best)
```

Three parts do the work:

1. **Diverse candidate sampling.** Temperature or nucleus on the planner distribution, or beam-diverse decoding. Duplicates waste rollouts.
2. **Action-conditioned world model.** A causal, few-step video world model (e.g. an interactive one distilled from a bidirectional teacher — see [causal-video-distillation.md](causal-video-distillation.md)) simulates each candidate for `H` steps. Fidelity is the ceiling on the search's value.
3. **Trajectory scoring.** Language-conditioned reward model, VLM classifier, or a small learned critic. Cheap rerank on imagined visual frames.

Compute allocation is dynamic: consequential branches (higher variance across candidates, further from training distribution) get more rollouts.

## Why it matters

Applies the o1-style test-time compute recipe to a domain where inference-time capability scaling had been muted. If the world model is accurate enough to be a search oracle, VLA quality can be scaled at deployment by rolling more, not by retraining — a substantially cheaper axis than more data or more parameters.

The pattern also puts world-model quality onto the critical path for embodied foundation models — previously world models were mostly a training-data ingredient (dynamics learning, curiosity). Now they matter at every inference decision.

## Gotchas & tricks

- **World-model drift dominates the ceiling.** A world model with 5% per-step error compounds to noise inside a horizon of ~20 steps; realistic budgets are shallow. Match `H` to fidelity.
- **Candidate diversity, not quality.** N identical high-probability subtasks give zero search benefit. Diverse decoding matters more than aggressive top-p at the planner.
- **When to bail.** Search is wasted when all candidates score identically. A "no-lift" heuristic (variance-threshold gate) short-circuits routine steps and concentrates budget on tough ones.
- **Reward-model gaming.** If the scoring VLM is a learned critic, aggressive candidates can Goodhart it. Prefer models that also gate on task-completion or physical plausibility, not just goal-embedding similarity.
- **Latency budget.** Real-time robots have hard deadlines. Prefer amortized search (bigger horizons less frequently) over per-step search that misses the control loop.

## Sources

- Paper: *τ₀-VLA: A Hierarchical Robot Foundation Model with World-Model-Guided Test-Time Computation* — 40-author team, 2026 — [arXiv:2608.16885](https://arxiv.org/abs/2608.16885).
- Related: *OpenAI o1 system card* — text-domain test-time compute; same "scale compute per decision" spine.
- Related: MPC and model-predictive control literature — the classical control analog of world-model-guided rollout selection.
