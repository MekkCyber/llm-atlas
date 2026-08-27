# Recursive Self-Improvement (Agents)

*Taxonomy — approaches to letting an agent system iteratively improve its own reasoning or execution stack.*

**TL;DR:** Recursive self-improvement (RSI) is the family of agent designs where a system *modifies its own behavior* over time based on experience. The central design tension is stability vs. depth: freeze the meta-layer and recursion caps at 1; let it edit itself and it destabilizes past ~2. Every current RSI approach fixes *some* part of the meta-layer and recurses on the parts it can safely change (memory, inputs, layer stack). The modern default in 2026 is to fix the meta-*operator* and change what it reads — Meta^n and Recuris are two concrete instances at different granularities.

**Related taxonomies:** —
**Depth files covered here:** [experiential-working-memory](experiential-working-memory.md) · [meta-n](meta-n.md)

---

## The problem

An agent that only ever runs its base policy plateaus on any task the base policy can't already solve. To improve *during deployment*, the agent needs to update *something* — its memory, its skills, its scaffolding, its reasoning strategy — based on what happened. But letting the system edit itself unrestricted destabilizes: the next update depends on the last, small errors compound, and the system drifts or oscillates.

Every RSI approach picks (a) what to fix, (b) what to change, and (c) how to prevent the change from destroying stability.

## The shared pattern

All approaches share three components:

1. **A fixed meta-layer** — the machinery that *proposes* updates. Its logic doesn't change across iterations.
2. **A changing state** — what the meta-layer reads and writes: memory, layer stack, skill library, scaffolding, reasoning strategy.
3. **A validation gate** — a check that the proposed update actually improves task success before landing.

The design decisions are *what to fix* and *what state to iterate on*.

## Variants

| Technique | Fixed meta-layer | Iterated state | When it wins |
| --- | --- | --- | --- |
| [experiential-working-memory](experiential-working-memory.md) | Meta-Agent (fixed prompt) | Skill Memory + Working Memory | Long-horizon tool-use tasks where skill selection is the bottleneck |
| [meta-n](meta-n.md) | Meta-operation Ω | Stack of pre-processes + helper libraries | Reasoning-heavy tasks with novel structure (e.g. ARC-AGI-2) |
| Voyager-style skill libraries (no depth file yet) | Skill-writer prompt | Skill library | Open-world exploration with reusable procedural skills |
| Self-editing agents (no depth file yet) | Kernel of "what can't be edited" | Everything else | Rare; usually collapses past depth 2 |

## How to choose

Pick by *where* the failure mode lives:

- **Failures live in skill selection / memory management** → experiential-working-memory. Fix the Meta-Agent, iterate the skill library and working-memory schema.
- **Failures live in reasoning strategy or task decomposition** → Meta^n. Fix Ω, iterate the layer stack that transforms problem input before it hits the solver.
- **Failures live in library breadth (missing reusable operations)** → Voyager-style skill library. Fix the skill writer, iterate the library.

The choices compose: a system can have Voyager-style skill accumulation *inside* an experiential-working-memory shell.

## Adjacent but distinct

- **Post-training / RL** — updates model *weights* from experience, not the harness or memory. Slower loop, permanent change. RSI targets the layer above.
- **Prompt optimization** (DSPy-style) — offline search over prompts against a fixed metric. No online change; no recursion depth in deployment.
- **Multi-agent orchestration** — running multiple agents in parallel. Doesn't recursively deepen a single system.

## Sources

- Paper: *Recursive Experiential-Working Memory Evolution for Long-Horizon Agent Harnesses* — Yu et al., 2026 — Recuris.
- Paper: *Meta^n: Recursive Self-Improvement through Emergent Depth* — Kim et al., 2026 — fixed Ω, input-recursive.
- Paper: *Voyager* — Wang et al., 2023 — early skill-library-focused RSI in Minecraft.
