# AutoMem

*Depth — automated two-loop training of memory management as a first-class cognitive skill for LLM agents.*

**TL;DR:** AutoMem promotes file-system operations (`write`, `read`, `edit`, `list`) to first-class actions the agent can pick alongside task actions, then trains *both* the surrounding memory structure (prompts, file schemas, action vocabulary) and the agent's proficiency at using it. Optimizing memory alone — without touching task-action behavior — improves a 32B open-weight agent 2–4× on Crafter, MiniHack, and NetHack, closing much of the gap to Claude Opus 4.5 and Gemini 3.1 Pro Thinking.

**Prereqs:** [../post-training/rejection-sampling.md](../post-training/rejection-sampling.md), [_agent-memory.md](_agent-memory.md)
**Related:** [../post-training/rlvr.md](../post-training/rlvr.md), [../post-training/_post-training.md](../post-training/_post-training.md)

---

## What it is

Most agent frameworks bolt a memory module onto a fixed task policy: retrieval, summarization, or a chat-history buffer glued to the outside. AutoMem instead treats "manage memory" as a **skill the agent chooses to exercise**, encoded as file-system actions that live in the same action space as the domain actions. The claim: memory competence is separable from task competence, and separating them lets each be optimized at its own timescale.

## How it works

### Two nested optimization loops

- **Outer loop (structure).** A strong LLM reviewer reads full trajectories and iteratively rewrites the *scaffold* that supports memory use: system prompts, file schemas ("what does a `plan.md` look like"), and the action vocabulary. The reviewer is looking for structural failures — the agent had the right idea but the schema didn't let it express it, the vocabulary is missing a `note.md` action, etc.
- **Inner loop (proficiency).** The agent's *good* memory decisions across many episodes are mined and become training signal — a rejection-sampling-style filter over memory actions specifically, not over task actions. The agent's memory proficiency sharpens without directly changing how it plays the game.

### Why two loops (and not one)

Trajectories run thousands of steps. A single bad memory action can hide until it corrupts a later decision. If you optimize structure and proficiency together, you can't tell whether "the agent got better" came from a smarter schema (structure) or better use of the same schema (proficiency). Decoupling gives each axis a clean gradient.

### Concrete action set

The agent's action space is `task_actions ∪ {write_memory, read_memory, edit_memory, list_memory, delete_memory}`. Each memory action takes a file path and content or query. The action masks change per environment but the memory verbs are invariant across Crafter, MiniHack, NetHack.

## Why it matters

- **Memory is the biggest lever for long-horizon agents.** Task-action policies plateau; the failures at horizon >1000 are almost all memory failures (lost plan, forgot early observation, re-derived same conclusion).
- **The gap between open-weight and frontier is largely a memory-competence gap.** A 32B open-weight agent using AutoMem approaches Claude Opus 4.5 / Gemini 3.1 Pro Thinking on long-horizon procedural games — an ~2–4× jump vs the baseline schema.
- **Separates training concerns.** You can now iterate on memory policy without breaking the task policy, and vice versa. That reshapes how agent training pipelines are structured.

## Gotchas & tricks

- **Reviewer quality caps outer-loop gains.** The scaffold rewriter is a strong LLM reading long trajectories — a weak reviewer stalls the outer loop even if the inner-loop signal is fine.
- **Rejection sampling of memory actions needs a way to *credit* them.** A `write` action at step 200 that saves the day at step 2000 has to be attributed correctly. AutoMem's inner loop uses episode-level outcomes; finer credit assignment is open work.
- **Watch for file-system reward hacking.** The agent can learn to write junk that flatters short-horizon reward without helping. Cap file counts, penalize unread writes.
- **Schema churn hurts stability.** If the outer loop rewrites the schema every episode, the inner loop's training data goes stale. Freeze the schema for N episodes between rewrites.

## Sources

- Paper: *AutoMem: Automated Learning of Memory as a Cognitive Skill* — 2026 — [arXiv:2607.01224](https://arxiv.org/abs/2607.01224).
- Environments: Crafter, MiniHack, NetHack (procedurally generated long-horizon games).
