# Agent Skills
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Reusable procedural units that agents load, discover, and compose at inference time. Each skill packages a description, invocation contract, and optional supporting files. Popularised by Anthropic's Claude Skills but now a common runtime primitive for frontier agents. The scaling problem: providing skills is easy; getting models to identify, apply, and coordinate them well requires targeted training data.

**Prereqs:** none (helpful: general agent-loop familiarity)
**Related:** [../post-training/fine-tuning/README.md](../post-training/fine-tuning/README.md), [skill-generation-rl](skill-generation-rl.md)

---

## What it is

A **Skill** is a self-contained bundle exposing a specific capability to an LLM agent, packaged so:

- The agent can **discover** it (typically via a short natural-language description registered in the agent runtime).
- The agent can **load** its full instructions on demand (avoiding always-in-context bloat).
- The agent can **invoke** deterministic side files (scripts, references, templates) that the skill ships with.

Skills sit between "tools" (single-call, schema-typed function invocations) and "prompts" (raw natural-language instructions). Concretely, a skill is usually a directory with a `SKILL.md` manifest plus references and scripts.

## How it works

Anatomy of a typical skill:

```
skill-name/
  SKILL.md               # frontmatter (name, description) + workflow
  references/            # docs the agent reads on demand
  templates/             # canonical outputs / prompts
  scripts/               # deterministic helpers (bash, Python)
```

At runtime:

1. **Registration** — the agent runtime scans a skill directory and exposes the (name, description) pairs as available skills.
2. **Discovery** — when a user request matches a skill description, the runtime injects an invocation notice into the LM's context (e.g. `Skill: xlsx is applicable to this task`).
3. **Loading** — the model calls `Skill(name)`, and the runtime injects the skill's full `SKILL.md` into context. Bulky reference files stay on disk until the skill explicitly reads them.
4. **Composition** — a skill can call other skills, use tools, or invoke its own scripts, giving deterministic subroutines inside model-driven work.

## Why it matters

- **Progressive disclosure.** Skills lift capability without permanent context cost. Only the description sits always-in-context; the body loads on demand.
- **Reuse.** A single skill file becomes the source of truth for a workflow used across many sessions.
- **Extensibility as a first-class primitive.** Skills unify tool-libraries, prompt-libraries, and few-shot exemplars into one addressable resource.
- **Trainable capability.** Skill-use is a distinct, measurable ability — worth targeting with dedicated SFT data (see [../data/README.md](../data/README.md) and the SKT paper).

## Gotchas & tricks

- **Description quality is the discoverability bottleneck.** If the frontmatter description doesn't match the way users phrase requests, the skill never triggers. Write description text with real user phrasings in mind.
- **Skill drift.** Skills authored against one model version tend to under-perform on newer ones — regenerate periodically.
- **Don't overpack.** A `SKILL.md` longer than ~200 lines starts to compete with the task's own context; move body content into `references/` files the skill reads on demand.
- **Skills are not tools.** A tool call has a rigid schema and a single response. A skill is a policy the model executes with all its ambient reasoning — use tools for narrow deterministic operations, skills for workflows.
- **Skill combinations are non-trivial.** Models often don't spontaneously compose two skills that would together solve a task. Explicit "meta-skills" that name common compositions help.

## Sources

- Anthropic blog: *Introducing Claude Skills* — anthropic.com/news, 2025.
- Paper: *SKT: Skill-Use Training at Scale via Verified Synthetic Data Generation* — arXiv:2608.02287, 2026 — trains models to use skills effectively via verified rollouts.
- Paper: *Progressive Agent Skill Generation via Reinforcement Learning* — arXiv:2608.01678, 2026 — RL for generating skills automatically.
