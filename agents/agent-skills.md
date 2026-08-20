# Agent Skills
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A **skill** is a structured, retrievable package of knowledge (instructions, examples, code) that an LLM agent can invoke inside its context. Empirically, skills help mostly by acting as **procedural anchors** that stabilise otherwise noisy execution — not by injecting missing facts. Retrieval, not authorship, is the ceiling: as the skill pool grows, the actual-use precision of retrieval falls sharply.

**Prereqs:** [../post-training/_rl.md](../post-training/_rl.md)
**Related:** none yet in this folder

---

## What it is

An agent skill is a discrete artifact (a markdown file, a JSON snippet, or a code function) that describes *how to accomplish a class of tasks* and is surfaced to the LLM at inference time via retrieval or explicit invocation. Popularised by Claude Skills, Voyager-style skill libraries, and Workflow Memory.

A "skill" is not just a prompt: it has three moving parts.

1. **Representation** — how the skill is written (imperative steps, code, example trajectory).
2. **Retrieval** — how the agent finds the right skill in a growing library.
3. **Adaptation** — how the agent instantiates the skill to the current task.

## How it works

At each step the agent:

1. Formulates a query from its current state.
2. Retrieves *k* candidate skills from a library (dense or sparse index over titles/descriptions).
3. Injects them into the LLM context, optionally as system-prompt cards.
4. Executes, then optionally writes back a new/updated skill on success.

Jiang et al. (2026) run a controlled ablation across 8,135 trials, then open-code 240 trajectory pairs into a **twelve-mode taxonomy** of skill/agent interactions grouped as:

- **Skills work** — procedural anchoring (dominant), tool binding, error recovery.
- **Skills fail** — brittle assumption, context mismatch, insufficient adaptation.
- **Skills are neutral / harmful** — distractor pull, confabulated invocation.

## Why it matters

Every self-evolving-agent framework (Voyager descendants, Claude Skills, SkillWeaver) assumes the library-growth story: keep adding skills and the agent gets better. This paper shows that only ~⅔ of successful uses look like "the skill anchored a shaky procedure"; only ~4.5% look like "the skill injected missing knowledge". Adding skills without fixing retrieval or anchoring makes things *worse*, not better.

Concrete numbers from Jiang et al. (2026):

- Skills beat Workflow Memory by **+6.06** points in matched comparisons.
- Procedural anchoring accounts for **65.7%** of successful skill uses; explicit knowledge injection **4.5%**.
- Actual-use precision falls from **29.6% → 3.3%** as the skill pool grows from **5 → 100**.

## Gotchas & tricks

- **Retrieval is the ceiling, not the library.** Optimise the ranker, not the authoring pipeline.
- **Exact-match invocation is neither sufficient nor necessary.** Downstream success is stable under confusable distractors as long as the anchor is procedurally similar.
- **Beware silent regressions.** A skill can hurt on tasks it wasn't written for by dragging attention off action-critical context — measure per-regime, not just aggregate success.
- **Anchor over inject.** Writing skills as *procedures* (numbered steps, pre/post conditions) beats writing them as *knowledge dumps*.

## Sources

- Paper: *Demystifying Agent Skills: Why They Work — Until They Don't* — Jiang, Huang, Xing, Wu, Gao, Cao, Wang, Liu, Li — UC San Diego / Princeton, 2026 — https://arxiv.org/abs/2608.14036
- Related: Claude Skills documentation, Voyager (Wang et al., 2023), Workflow Memory literature.
