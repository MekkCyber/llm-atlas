# Experience-Handoff RL
*Depth — co-training an agent as actor, reflector, and playbook curator under a shared downstream reward.*

**TL;DR:** Scalar-reward RL for agents updates only the actor's token probabilities and throws away the rich environment feedback (error logs, intermediate states) collected during a rollout. **Experience-Handoff RL** fixes an agentic scaffold with a *playbook* artifact and trains the same model under three roles — **action generator**, **insight reflector**, and **playbook curator** — against one shared downstream reward: the effect of the written playbook on future rollouts' actor performance. The agent learns not only what to do, but what to write down for its future self.

**Prereqs:** [../post-training/rlvr.md](../post-training/rlvr.md), [../post-training/grpo.md](../post-training/grpo.md)
**Related:** [../systems/partial-rollouts.md](../systems/partial-rollouts.md), [../post-training/_rl.md](../post-training/_rl.md)

---

## What it is

A training recipe for agents that have a **persistent memory artifact** (a playbook, a notes file, a strategy document) that future rollouts inherit. The artifact is the handoff; the recipe trains the agent to construct it well.

Three roles, same model:

- **Action generator:** picks actions step by step during the rollout. Standard actor.
- **Insight reflector:** at episode end (or periodically), writes reflections on what happened — what worked, what failed, what pattern to remember.
- **Playbook curator:** merges reflections into the persistent playbook artifact. Decides what stays, what gets rewritten, what gets deleted.

All three run inside a fixed scaffold — the paper uses the Agentic Context Engineering (ACE) harness — so the handoff mechanics are held constant and only the model's policy is trained.

## How it works

### Joint training objective

Standard scalar-reward RL updates $\pi_\theta$ to maximize

$$
J_\text{act}(\theta) = \mathbb{E}_\tau [R_\text{terminal}(\tau)]
$$

Experience-Handoff adds a downstream term: the playbook $P$ written at the end of task $i$ should improve *actor performance on future tasks*:

$$
J_\text{ho}(\theta) = \mathbb{E}_i [R_\text{terminal}(\tau_{i+1}(P_i))] - \mathbb{E}_i [R_\text{terminal}(\tau_{i+1}(\emptyset))]
$$

The gradient is still scalar-reward RL (GRPO-style), but the *objects* being optimized include the reflection and curation outputs, not only the actions. Credit assignment uses the downstream effect of each handoff artifact on future rollouts' rewards.

### Role-specific prompts, shared weights

The three roles are prompt-conditioned on the same backbone. Different system prompts select the role:

- "You are the actor…" → action generator.
- "You just finished the task. Reflect…" → reflector.
- "Update the playbook with this reflection…" → curator.

Shared weights are updated by whichever role produced the token in the rollout that got credit. In practice this means curator tokens get updated more slowly (sparse reward attribution through future rollouts) than actor tokens.

### Why the scaffold is fixed

The ACE harness defines when reflection happens, how the playbook is formatted, and how it's exposed to future actors. Fixing it means the gradient signal isn't contaminated by scaffold search — the paper is isolating the question "can the model's artifact-writing improve from RL."

## Why it matters

- Pushes agent self-improvement beyond trajectory-level reward: the model's own scaffolding is now a training target.
- The pattern (optimize the handoff artifacts, not only the actions) generalizes to any agent system with persistent memory, notes, or an inherited plan.
- Reports higher held-out agentic performance than scalar-reward baselines that update only the actor — evidence that the handoff tokens carry useful gradient signal.

## Gotchas & tricks

- Credit assignment is sparse: a curator token can only be credited by the next rollout's reward, which is both delayed and noisy. Use batch-level variance reduction (group baselines) aggressively.
- Playbook format has to constrain the curator's output — unconstrained prose tends to grow into sprawling notes and dilute itself. Templated bullet lists work better than free text.
- The "fixed scaffold" simplifies attribution but hides a question: a learned scaffold (reflection timing, playbook schema) is probably the next step.
- Reward is from a verifier on *future* rollouts, so there is no way to avoid running those rollouts. Compute cost per training step is higher than vanilla GRPO.

## Sources

- Paper: *Experience-Handoff Reinforcement Learning for Recursive Self-Improvement* — Kim, Jeon, Zhang, Yang, 2026 — https://arxiv.org/abs/2611.16409
- Related: *Reflexion: Language Agents with Verbal Reinforcement Learning* — 2023 — early reflection-based scaffold.
