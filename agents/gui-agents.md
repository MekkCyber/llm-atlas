# GUI Agents
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A **GUI agent** is an LLM-driven system that executes tasks by driving graphical user interfaces — clicking buttons, typing into fields, scrolling, waiting for pop-ups — usually on mobile, Android, or desktop OS simulators. The execution loop is longer than a chat turn, the state space is a dynamic screen tree, and plans invalidate whenever the UI shifts. The pragmatic recipe today is **training-free skill evolution**: keep a live library of skills that the agent revises in-place from execution feedback, rather than continuously re-tuning weights.

**Prereqs:** [README.md](README.md), [coding-agent-harness.md](coding-agent-harness.md)
**Related:** [../multimodal/README.md](../multimodal/README.md)

---

## What it is

GUI agents differ from tool-calling agents in three ways:

- **Perception** is grounded in screenshots or accessibility trees — a multimodal input for every step.
- **Actions** are (element, action-type) pairs that must be resolved against a UI that may have relocated, resized, or replaced the target since the plan was made.
- **Trajectories are long** — a typical benchmark task spans dozens of steps, with waits and reloads that a chat harness would not tolerate.

The canonical benchmark suite is **MobileWorld / AndroidWorld / OSWorld**, each with an interactive OS simulator that scores task completion.

## How it works

Training-free skill evolution ("Reflect, Revise, Reuse") illustrates the pattern:

1. **Skill library** — a set of parameterized procedures (`login(app)`, `search_and_open(query)`), stored as editable files.
2. **Reflect** — on failure, an isolated critic diagnoses the trajectory: which step went wrong, and whether the failure was in the skill or the plan.
3. **Revise** — the agent uses a restricted tool interface (edit_skill, add_step, add_precondition) to patch the skill file directly. No gradient updates.
4. **Reuse** — the revised skill is available for related tasks in the same session and, once verified, promoted to the shared library.

Because revisions are code-level edits rather than weight updates, they transfer across tasks that touch the same UI subsurface and can be inspected or rolled back by a human.

## Why it matters

- **Retraining is impractical.** UIs change constantly (app updates, A/B tests, OS versions); a GUI agent trained on a fixed snapshot degrades within weeks.
- **Training-free adaptation** keeps the agent functional across UI drift without a fine-tuning pipeline behind it.
- **Skill libraries are auditable** — humans can read, edit, or gate individual skills before deployment.

Empirically the reflect-revise-reuse loop lifts task completion by **+16.2%** on MobileWorld, **+6.0%** on AndroidWorld, and **+10.5%** on OSWorld across multiple base models, and the evolved skill libraries transfer to unseen but related tasks.

## Gotchas & tricks

- Skill files grow if not garbage-collected — cap library size and prune skills that fail more than they succeed.
- Critic isolation is essential: reusing the executor as its own critic biases both roles toward the same failure modes.
- Reflect-revise cycles inside a single task waste tokens — restrict them to failure boundaries, not every step.

## Sources

- "Reflect, Revise, Reuse: Training-Free Skill Evolution for GUI Agents" — Chen et al., 2026 — [arXiv:2609.17653](https://arxiv.org/abs/2609.17653).
