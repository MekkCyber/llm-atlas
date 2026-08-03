# Agent Data Flywheel
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Agent training data is expensive — trajectories are long, environments are messy, and hand-labelling doesn't scale. A **data flywheel** pattern automates the whole loop: agents themselves *synthesize tasks*, *inject environment states*, *write verifiers*, *judge trajectories at step level*, and *analyze failures* to prioritize the next round. Qwen-UI-Agent's **AutoResearch** flywheel (2026) generates ~10,000 validated task–verifier pairs per Online RL round with no human in the loop.

**Prereqs:** [gui-agents](gui-agents.md), [../post-training/rlvr](../post-training/rlvr.md)
**Related:** [../case-studies/qwen-ui-agent](../case-studies/qwen-ui-agent.md), [../post-training/rl-prompt-curation](../post-training/rl-prompt-curation.md)

---

## What it is

Every agent RL loop needs three ingredients: **tasks** (prompts the agent tries), **environments** (states the agent starts in), and **verifiers** (functions scoring whether the agent succeeded). Hand-writing all three is a career-length project. A data flywheel closes the loop: agents themselves generate the ingredients, using the current model as the labeller for the next model.

## How it works

The Qwen-UI-Agent AutoResearch flywheel has five stages:

**1. Task synthesis.** Hierarchical *function trees* enumerate what each app/tool can do; *capability profiles* enumerate what the current policy already handles. Task-synthesis agents combine the two into fresh tasks that target uncovered capabilities.

**2. Environment State Synthesis.** Coding agents analyze the sandbox codebases (Android app source, web-app fixtures, Ubuntu VM setup scripts) and distill *reusable data-injection skills* — programmable ways to seed the environment into arbitrary starting states (e.g. "populate the inbox with 42 unread emails from these senders").

**3. Verifier synthesis.** For each task, an agent writes a task-specific code-based verifier — a function that reads the final environment state and returns a scalar success signal. Verifiers are validated by cross-rollout: if all agent rollouts agree with the verifier, or a spot-check human confirms, it's kept.

**4. Step-Level VLM Judge.** A vision-language judge processes each trajectory and extracts three supervision signals:
- Maximal contiguous *correctly-advancing* step spans (positive SFT data).
- The first step that *initiates reflection or exploration* (planning signal).
- *Recovery segments* returning from an erroneous state to a valid path (repair signal).

**5. Failure analysis.** An analysis agent examines each failed trajectory and partitions failures into **model / environment / task / verifier** failures. Model failures map to a structured cause (application knowledge gap, constraint violation, state-tracking error). The output feeds Task Synthesis in the next round, prioritizing the model's weaknesses.

## Why it matters

- **Scales without human labelling.** ~10,000 validated task–verifier pairs per RL round.
- **Adaptive.** Failure analysis makes each round target the current model's weaknesses — capability grows where the flywheel spins.
- **Reproducible RL environments.** Environment-state synthesis via code (not screenshots or human replay) means the same state can be seeded thousands of times deterministically.
- **Applies beyond GUI agents** — the pattern (tasks + envs + verifiers, all agent-synthesized) transfers to code agents, browser agents, and scientific-agent stacks.

## Gotchas & tricks

- **Verifier drift.** Agent-written verifiers can silently drift toward measuring what the model is good at, not what the task requires. Cross-rollout validation and human spot-checks matter.
- **Task diversity collapse.** Task synthesis converges on a small mode if it's not explicitly diversified (temperature, forced coverage of function tree, novelty rewards).
- **Environment-injection skills are the leverage** — most of the wall-clock win comes from being able to seed arbitrary states cheaply.
- **Step-level judging needs a strong VLM** — a weak judge injects noise faster than tasks add signal.
- **Failure taxonomies must be pre-defined.** "Model failure" is not a useful bucket; break it down into causes that map to actionable data-synthesis operators.

## Sources

- Paper: *Qwen-UI-Agent Technical Report* — MAI-UI Team, Alibaba, 2026 — [arXiv:2607.28227](https://arxiv.org/abs/2607.28227).
- Precursor: automated task/environment generation in code-agent stacks (SWE-Gym-family, self-generated evaluation benchmarks).
