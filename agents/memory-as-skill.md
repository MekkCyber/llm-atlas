# Memory as a Learnable Skill
*Depth — treating an agent's memory management as a trainable policy, not a hand-designed system.*

**TL;DR:** Long-horizon agents fail because their memory is hand-coded (fixed prompts, fixed retrievers) and neither the memory *schema* nor the model's *use of it* gets to improve from data. **AutoMem** treats memory management as a first-class skill: promote filesystem operations (read/write/list/append) to actions the policy chooses among, then optimize in two loops — an outer loop that rewrites the memory schema after inspecting whole trajectories, and an inner loop that fine-tunes the policy on its own good memory decisions. On Crafter, MiniHack, and NetHack, optimizing memory alone (no changes to task-action behavior) improves a 32B open model 2–4×, matching frontier closed models.

**Prereqs:** [../post-training/rejection-sampling.md](../post-training/rejection-sampling.md)
**Related:** [../post-training/_post-training.md](../post-training/_post-training.md), [../post-training/reasoning/long-cot-rl.md](../post-training/reasoning/long-cot-rl.md)

---

## What it is

Most memory-augmented agents (ReAct, Voyager, Reflexion) hard-code both the *storage schema* (what files, what keys, what fields) and the *retrieval policy* (when to look, what to fetch, when to write). These choices are made by humans up front and never trained. When they're wrong — the schema misses a field the task needs, the policy retrieves at the wrong time — the agent silently degrades over long horizons.

"Memory as a skill" reframes memory management as a *learnable action space*. The model chooses among file-system-like operations (list, read, append, overwrite, delete) at each step, alongside its task actions. The memory it produces is judged by whether it downstream helps future decisions, and the whole loop trains.

## How it works

Two nested optimization loops:

**Outer loop — schema rewriting.** A strong LLM (the "coach") is given complete agent trajectories after each episode batch. It rewrites the *structural context* of the memory system: the prompts telling the agent what memory is for, the file schemas defining what to store, the action vocabulary defining what operations exist. This loop is expensive but runs at a low cadence.

**Inner loop — policy fine-tuning.** From the same trajectories, identify the agent's *good* memory decisions — ones that were later retrieved and helped succeed. These become supervised training data to sharpen the policy's memory proficiency directly (SFT or preference-style updates). This loop runs often.

The two loops interact: a better schema makes better decisions easier; better decisions supply better training data; both together compound.

## Why it matters

- **Memory is high-leverage.** On Crafter / MiniHack / NetHack, optimizing memory *only* — never touching how the agent takes task actions — improves a base agent 2–4×. Memory turns out to be where a lot of the failure mass lives, and it's independently learnable.
- **Closes the frontier gap for open models.** With AutoMem, a 32B open model becomes competitive with Claude Opus 4.5 and Gemini 3.1 Pro Thinking on the same long-horizon games — a striking result given the parameter and pretraining-data gap.
- **Schema and policy must co-evolve.** Fixing either half at hand-designed defaults loses most of the gain. The paper's ablations show both loops contribute complementary improvements.

## Gotchas & tricks

- **The coach LLM needs to see the whole trajectory** — thousands of steps in games like NetHack. Trajectory review is the dominant cost of the outer loop; batch aggressively and use trajectory summaries where possible.
- **Reward signal is sparse and delayed.** A memory write's value is only visible when it's later retrieved and helps. Standard credit-assignment tricks (return normalization, discounting) apply.
- **Not the same as RAG.** Retrieval-augmented generation stores external, static knowledge; here the agent is authoring, indexing, and consulting its own working memory of the current episode.
- **Compose with task-action training.** The paper shows memory-only gains; the natural next step is joint task+memory RL, at higher cost.

## Sources

- Paper: *AutoMem: Automated Learning of Memory as a Cognitive Skill*, 2026 — [arXiv:2607.01224](https://arxiv.org/abs/2607.01224).
- Related: *Voyager: An Open-Ended Embodied Agent with LLMs* — Wang et al., 2023 — hand-designed skill library predecessor.
- Related: *Reflexion: Language Agents with Verbal Reinforcement Learning* — Shinn et al., 2023 — hand-designed reflection memory.
