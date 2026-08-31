# Skill evolution via persistent wiki
*Depth — co-evolving agent skills with a curated knowledge base (WikiSkill).*

**TL;DR:** Automatic skill discovery from agent experience keeps the *procedures* (executable skills) but throws away the *insights* behind them, so later iterations can't build on earlier learning. WikiSkill splits agent memory into three tiers — raw experience, an accumulated **wiki** of general knowledge, and executable skills — and iterates all three together. Skills evolve on top of a stable, human-readable wiki rather than an ever-growing raw log.

**Prereqs:** [README.md](README.md)
**Related:** [skill-retrieval.md](skill-retrieval.md), [live-self-improvement.md](live-self-improvement.md)

---

## What it is

A framework for turning agent runs into persistent capability. Instead of appending every trajectory to a monolithic memory or a flat skill library, WikiSkill maintains three separate stores:

1. **Raw experience** — trajectories, tool traces, rewards. Ephemeral working memory.
2. **Wiki** — accumulated general-purpose articles distilled from experience (concepts, gotchas, environment quirks).
3. **Skills** — executable procedures the agent invokes at runtime.

Skill updates read the *wiki*, not the raw log; the wiki refactors and generalizes; raw experience is consumed and largely discarded.

## How it works

Per iteration:

1. Run the agent on a batch of tasks; collect trajectories and reward signals.
2. Distill new / updated wiki articles from those trajectories (LLM summarizer conditioned on prior wiki state to avoid duplication).
3. Update skill definitions using the *new wiki state* as context — skills are refined against generalized knowledge rather than idiosyncratic runs.
4. Retire or merge obsolete skills; the wiki carries the institutional memory forward even if a specific skill is deleted.

At inference, the agent retrieves relevant skills (and, optionally, wiki articles for context).

## Why it matters

- Skills evolved with a wiki tier **outperform prior skill-evolution methods** across diverse benchmarks and backbones.
- Smaller models + evolved skills often beat larger models without them — evidence that memory quality substitutes for parameter count in agentic settings.
- Skills transfer across model families; skills evolved by one model can beat self-evolved skills for another.
- Ablations confirm the wiki is the critical tier; removing it collapses gains to baseline.

## Gotchas & tricks

- Wiki bloat is real — enforce a max-articles budget with LLM-driven merges, or article quality degrades.
- Distillation must be conditioned on prior wiki state to avoid re-summarizing the same lessons every iteration.
- Skill retirement is often skipped in practice; without it stale skills poison retrieval.
- Distinguish from vanilla RAG: the wiki is *rewritten*, not just appended; and skills are executable, not passages.

## Sources

- Paper: *WikiSkill: Compiling Agent Experience into Persistent Knowledge for Skill Evolution* — Tang, Rashtchian, Ferng, Tomkins, Juan, Vu, 2026 — [arXiv:2608.27454](https://arxiv.org/abs/2608.27454)
