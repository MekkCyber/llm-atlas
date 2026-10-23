# Agent Memory

*Taxonomy — persistent state for LLM agents across turns, sessions, and tasks.*

**TL;DR:** Foundation-model agents have fixed weights and bounded context, so sustained learning from experience requires an **external memory** layer. The usual framing treats memory as a retrieval cache judged by recall. A more useful framing judges it by whether having (vs. not having) the memory *changes the next action*. The space factorizes into three axes — **substrates** (how it's stored), **assets** (what's stored), **policies** (what the agent does with it) — with a two-loop lifecycle (write loop + read-and-act loop).

**Related taxonomies:** [_post-training](../post-training/_post-training.md), [_rl](../post-training/_rl.md)
**Depth files covered here:** [socratic-skill-learning](socratic-skill-learning.md) · [agent-kv-cache](../inference/agent-kv-cache.md)

---

## The problem

An agent that cannot change behavior based on past experience is a chatbot. But model weights are frozen (training is slow) and the context window is bounded (every turn starts near-empty for cold sessions). Memory is the layer that spans sessions: it accumulates what has been tried, what worked, which tools are reliable, what the user prefers.

The research mistake has been scoring memory on *information retrieval* — "did you remember fact X?" — rather than on the thing that matters: *does the agent act better next time*. The retention/retrieval benchmarks that dominate the literature can be aced by systems that help the agent not at all.

## The shared pattern

Every agent-memory system has the same two-loop structure:

1. **Write loop.** After (or during) a task, decide what to persist. Options range from raw transcripts to summarized spans to parametric updates.
2. **Read-and-act loop.** On the next task, decide what to retrieve, how to inject it (prompt, in-context, parameter patch), and whether the retrieved content changes the action.

What varies is the three axes below.

## Variants

| Technique | Key idea | Main tradeoff | When it wins |
| --- | --- | --- | --- |
| [socratic-skill-learning](socratic-skill-learning.md) | Student writes reusable Skill documents while learning | Human-inspectable, bounded size, transfers across sessions | Repeated families of sub-tasks |
| Retrieval-augmented memory | Store raw/summarized turns; dense-retrieve on new query | Simple to build; can be gamed by recall benchmarks | Open-domain QA over past sessions |
| Parametric memory (LoRA / adapter patches) | Encode experience into adapter weights | Dense but opaque; hard to inspect or revise | High-volume task-specific personalization |
| [agent-kv-cache](../inference/agent-kv-cache.md) | Structured, editable KV cache within a trajectory | Fast within-session; does not persist across sessions | Long single-session agent runs |
| Episodic buffers (no depth file yet) | Store full past trajectories, retrieve by task similarity | Rich but expensive; retrieval drift | When sub-task matching needs full-episode context |

Axes behind the table:

- **Substrate** — raw tokens / summaries / dense vectors / adapter weights.
- **Asset type** — facts, procedures (skills), preferences, outcomes.
- **Policy** — when to write, when to read, how to combine with current context.

## How to choose

Default to **text-based memory with inspectable artifacts** (Skills, summaries, structured facts) over parametric updates — human review is cheap and the memory itself can be edited. Reach for parametric memory only when the volume of experience is too large to carry in prompts and the task is well-defined enough that a wrong update is cheap to roll back. Within-session use **structured KV caches** rather than naïve rolling context.

Judge every design against the action-change metric: run the agent *with* and *without* its memory on the same held-out task, and measure the delta in success rate. Retention / lookup accuracy is a lower-bound proxy, not the goal.

## Adjacent but distinct

- **In-context learning** — scales within one prompt, not across sessions.
- **Prompt engineering** — static instructions, no update loop.
- **Fine-tuning / continual learning** — updates the base model's weights, not an auxiliary store.
- **RAG over a knowledge base** — retrieval over *external* corpora the agent didn't author.

## Sources

- Survey: *Memory-Centric Continual Learning for AI Agents* — HIT / Tsinghua / Huawei consortium, 2026 — [arXiv:2610.29357](https://arxiv.org/abs/2610.29357). The source of the three-axis taxonomy and the action-change evaluation framing.
- Paper: *Socratic Skill Learning* — Nanjing / Alibaba, 2026 — covers inspectable Skill artifacts.
- Paper: *Cache Engineering for Agents* — Tsinghua / CUHK, 2026 — within-session structured cache.
