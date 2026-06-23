# Hierarchical Agent Memory

*Depth — splitting agent memory into long-term / working / tool stores scoped by temporal lifetime.*

**TL;DR:** Most agent-memory implementations are a single retrieval-augmented blob. **Hierarchical agent memory** instead splits memory into three explicit stores with different write/read policies: **long-term memory** (persistent user profile + reusable tool experience), **working memory** (preferences active during the current multi-turn session), and **tool memory** (reusable execution experience for reliable repeatable actions). Each store is scoped by *temporal lifetime*, and the agent reads/writes to each at different points in the loop. Introduced as MemSlides (Jin et al., 2026) for personalized slide-generation agents.

**Prereqs:** [_agent-memory](_agent-memory.md)
**Related:** [memory-governance](memory-governance.md), [governed-memory](governed-memory.md)

---

## What it is

Three stores, three lifetimes:

| Store | Lifetime | Contents | Read | Write |
| --- | --- | --- | --- | --- |
| **User-profile memory** (long-term) | Persists across sessions | Intent-conditioned preferences, persona | Round 0 of every session | Periodic distillation from session activity |
| **Working memory** | Single multi-turn session | Active preferences, session constraints | Every turn | Each turn that introduces a constraint |
| **Tool memory** (long-term) | Persists across sessions | Reusable execution experience for tools / skills | When the agent picks a tool | After a successful tool execution worth memorizing |

Paired with a **scoped local-revision** policy: when a user asks for a modification, the agent locates the smallest affected region (e.g. one slide of a deck) and applies the change there, rather than regenerating the full artifact. This keeps the action surface narrow so tool memory entries are actually reusable.

## How it works

Per-session loop:

```
Session start:
  load user-profile memory   →   compute personalized "round 0" plan
  init empty working memory

Per turn:
  read user message
  update working memory if a new preference / constraint was introduced
  for each action:
      consult tool memory for matching execution traces
      execute the action on the smallest affected region only

Session end:
  optionally distill stable preferences from working memory back into user-profile memory
  optionally promote successful tool traces into tool memory
```

The key design choice is the *write policy* per store: working memory writes are eager, long-term writes are gated by stability or success — so transient session quirks don't pollute the persistent profile.

## Why it matters

The conflation failure mode — a single store mixing "what the user said this turn" with "what the user prefers in general" — is the dominant cause of personalization drift in multi-turn agent loops. Splitting by temporal scope means the agent can apply the right policy per write (eager / gated) and the right retrieval policy per read (recent / persistent / skill-matched). Reported wins on persona-alignment judgments and closed-loop revision behavior; the headline isn't a SOTA number but a clean decomposition that each component is shown to handle independently.

## Gotchas & tricks

- **Promotion criteria from working → long-term matter.** Promote too eagerly and you contaminate the profile with one-off requests; promote too conservatively and the agent never learns. The MemSlides paper uses qualitative criteria; production systems will want explicit thresholds.
- **Scoped revision needs accurate localization.** If the agent picks the wrong "smallest affected region," it produces inconsistent state. This is the dominant failure mode and is independent of the memory architecture.
- **Tool memory is closer to skill memory.** What you're storing isn't raw traces but distilled "this tool, with these args, does X" entries — see [governed-memory](governed-memory.md) for how to add quality/lifecycle metadata to keep this clean.
- **Pairs naturally with governance metadata.** This pattern is *temporal* hierarchy; layering [governed-memory](governed-memory.md) on top adds *quality* governance. They're orthogonal.

## Sources

- Paper: *MemSlides: A Hierarchical Memory Driven Agent Framework for Personalized Slide Generation with Multi-turn Local Revision* — Jin, Xu, Zhu, Yang, 2026 — https://arxiv.org/abs/2606.17162
