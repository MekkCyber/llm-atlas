# Agent memory
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** For long-horizon LLM agents, "memory" is a **contract about what each decision may see**. Two complementary axes: the **structure** (schemas, action vocabulary, retrieval typing) that constrains and shapes the memory, and the **proficiency** the model gains at using it. Recent work (**AutoMem**, 2026; **AgenticSTS**, 2026) shows memory is an independently trainable skill: optimizing it alone — with no changes to task-action behavior — can 2–4× long-horizon performance and bring open-weight models competitive with frontier systems.

**Prereqs:** [_rl](../post-training/_rl.md)
**Related:** [rejection-sampling](../post-training/rejection-sampling.md), [README](README.md)

---

## What it is

The default agent memory recipe is to append every observation, tool call, and reflection to the prompt. This is easy but has two big problems:

1. **Prompt grows without bound.** Long runs blow past context or hit degraded long-context regions.
2. **Ablation-hostile.** With everything jumbled into one transcript, no memory layer (episodic, semantic, reflective) can be individually attributed to a behavior.

The alternative framing (AgenticSTS): treat memory as a **typed retrieval layer** that assembles a bounded fresh prompt per decision. What each decision sees is a **contract**. AutoMem takes it further and treats memory as a **learned cognitive skill**: file-system operations become first-class actions the model can choose alongside task actions.

---

## How it works

**Two axes to optimize:**

- **Structure.** The prompt template, the retrieval-type schema (episodic, semantic, working-set), the file/schema layout, the memory-action vocabulary. AutoMem's outer loop uses a *strong* LLM to review whole trajectories and iteratively revise this structure — a form of prompt/schema search.
- **Proficiency.** The agent's own skill at using the structure — when to encode, when to retrieve, what to keep. AutoMem's inner loop mines good memory decisions from many episodes and trains on them directly (rejection-sampling-style behavior cloning), sharpening the model's memory policy without touching its task-action policy.

**Bounded contract (AgenticSTS).** Each decision starts from a *fresh* user message assembled by typed retrieval, with no raw cross-decision transcript appended. Prompt length stays bounded across arbitrarily long runs; each memory layer can be ablated in isolation.

**Empirical result.** On procedurally generated long-horizon games (Crafter, MiniHack, NetHack), optimizing memory alone lifts a 32B open-weight agent's success by **~2–4×**, matching frontier systems like Claude Opus 4.5 and Gemini 3.1 Pro Thinking.

---

## Why it matters

- **Isolable variable.** Fixed task-action policies + evolving memory schema/proficiency turns memory into a controlled experimental variable. Comparisons across memory designs become clean.
- **Non-parametric wins.** Structure edits (schema, action vocab, retrieval typing) are free at inference — no fine-tuning required. Proficiency training is cheap because good decisions can be labeled offline from trajectories.
- **Bounded prompts.** Long-horizon runs stop being a context problem — throughput, cost, and long-context degradation all improve simultaneously.
- **Complementary to task-action RL.** The memory axis has been left on the table by most agent-RL work. AutoMem's ablations suggest it may be the higher-leverage axis on long-horizon tasks than task-action fine-tuning.

---

## Gotchas & tricks

- **Retrieval typing must be authored.** You need a schema of memory types the agent can write into and query. Wrong typing (too narrow or too broad) bottlenecks the whole system. AutoMem's outer loop tunes this; without automated tuning, expect several manual iterations.
- **Structure/proficiency interact.** Training proficiency against a broken structure locks in a workaround. AutoMem alternates the two loops explicitly; sequential (structure first, then proficiency) tends to underperform.
- **Testbeds matter.** Closed-rule, stochastic long-horizon games (Slay the Spire 2, NetHack, MiniHack) make results reproducible; open-ended tasks make it hard to attribute gains to memory vs task policy.
- **Beware "long-context" as a memory replacement.** Larger context windows don't remove the ablation problem — they just delay it. Bounded typed retrieval + writable memory is a different design point.

---

## Sources

- Paper: *AutoMem: Automated Learning of Memory as a Cognitive Skill* — 2026 — [arXiv:2607.01224](https://arxiv.org/abs/2607.01224).
- Paper: *AgenticSTS: A Bounded-Memory Testbed for Long-Horizon LLM Agents* — Cheng et al., 2026 — [arXiv:2607.02255](https://arxiv.org/abs/2607.02255).
