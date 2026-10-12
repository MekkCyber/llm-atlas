# Code Worlds for Agent Training (AgentGarten)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A platform pattern that pairs **symbolic game engines or simulators** (owning state, physics, rules) with a **shared neural renderer** (a distilled real-time video model) that generates photorealistic observations from structured conditions. One perception stack generalises across arbitrary code worlds; experience is passed to successor agents as natural-language playbooks. Reports competitive task performance after 4 rounds of experience vs. millions of episodes for conventional RL.

**Prereqs:** [README.md](./README.md), [../post-training/_rl.md](../post-training/_rl.md)
**Related:** [language-world-models.md](./language-world-models.md)

---

## What it is

An agent-training platform where many diverse training environments share a single rendering layer. Each environment is a "code world" — a program (engine or simulator) authoritative over state, rules, and transitions. The renderer turns the environment's structured state description into pixels the agent perceives.

## How it works

- **Engine of record.** A code world keeps canonical state (positions, inventories, scores). Normal game-engine or simulator semantics; agents cannot violate rules because the engine enforces them.
- **Shared neural renderer.** A video diffusion model, distilled with the paper's *Adversarial Forcing* method so it runs in real time, converts each state description into a photorealistic observation. One trained renderer serves every code world; adding a new world does not require retraining the renderer.
- **Playbooks.** When a cohort of agents completes a world, their experience is compiled into a natural-language playbook — curriculum notes, strategies, pitfalls. Later cohorts condition on the playbook, inheriting hard-won structure without replaying all the episodes.

## Why it matters

The usual bottleneck in agent RL is environment coverage: either simulate many worlds (expensive to render each) or train one specialist per world (doesn't generalise). AgentGarten decouples physics-faithful simulation (cheap, procedural) from photorealistic rendering (one amortised model), and curricularises experience across cohorts. The 4-rounds-vs-millions-of-episodes comparison is the headline claim; whatever the exact replication, the architectural pattern is portable.

## Gotchas & tricks

- The renderer is a bounded contract: it must handle any structured state the code world emits. A code world that invents new object kinds without updating the renderer gets degraded output.
- Playbooks only transmit what the LLM-compiler can articulate. Non-verbal skills (motor patterns, tight timing loops) transfer poorly.
- Agent observations are real-time video; the renderer latency has to beat the engine's step budget to stay closed-loop.
- "Adversarial Forcing" is the renderer-distillation trick that gets diffusion into real-time; worth factoring out as its own method if it generalises.

## Sources

- Paper: *AgentGarten: Code Worlds for Evolving Agents* — Chi et al., MirroS / Tsinghua / PKU, 2026 — [arXiv:2610.12374](https://arxiv.org/abs/2610.12374)
