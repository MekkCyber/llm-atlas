# Hypothesis-Tree Refinement (HTR)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** An autonomous-research agent design where long-horizon experimentation is organized as a **persistent tree of hypotheses, artifacts, evidence, and distilled lessons**. A long-lived **coordinator** maintains the tree and decides which branches to expand; short-lived **executors** test individual hypotheses in isolated git worktrees so failed attempts don't pollute the main artifact. As executors return results, the tree is updated: verified improvements are admitted, failed branches leave behind distilled lessons, and the search frontier refines. Introduced in **Arbor** (2026); achieves >2.5× the held-out gain of Codex / Claude Code on six real research tasks and SOTA on MLE-Bench Lite.

**Prereqs:** [README.md](README.md)
**Related:** [../post-training/reasoning/mcts.md](../post-training/reasoning/mcts.md) · [../post-training/rl-prompt-curation.md](../post-training/rl-prompt-curation.md)

---

## What it is

Most "autonomous researcher" stacks are **flat ReAct loops**: the agent runs one attempt, observes the outcome, runs another attempt, and so on, with no structural memory across attempts. The agent forgets *why* an earlier branch failed by the time it picks a new direction. HTR adds *cumulative state*: every attempt becomes a node in a tree, every result feeds back into the tree, and the tree is the agent's working memory.

Three structural moves chained together:
1. **Persistent hypothesis tree** as the agent's working memory.
2. **Coordinator / executor split** — long-lived planner vs short-lived testers.
3. **Distilled lessons** that propagate from failed branches to prune sibling branches.

---

## How it works

### The tree

Each node is a **hypothesis** — a candidate research direction. Edges carry the **artifacts** (code, configs), **evidence** (metrics from running the hypothesis), and **distilled lessons** (natural-language summaries of what was learned). Nodes have status (untested, running, verified, rejected, dominated).

### Coordinator (long-lived)

A single agent process that:
- Maintains the global tree state.
- Decides which leaves to expand next (search policy over the tree).
- Reads distilled lessons from completed nodes to inform expansion choices.
- Admits verified improvements into the main artifact branch.
- Lives for the entire run, accumulating context.

### Executors (short-lived)

For each hypothesis to test, spawn a fresh agent in an **isolated git worktree**:
- The worktree has the current main artifact as its starting point.
- The executor implements and tests the hypothesis without seeing the broader tree.
- It produces: a modified artifact, evidence (metrics), and a natural-language lesson summary.
- It then terminates. Its context never grows long.

The worktree isolation is the key reliability trick: a failed or destructive executor can only damage its own worktree, never the main artifact or other branches.

### Lesson propagation

When an executor finishes, the coordinator distills the result into a **lesson** — a short natural-language summary like "approach X failed because of constraint Y" or "the bottleneck is Z, not the previously suspected W." Lessons attach to the parent node and are visible to all future expansion decisions in that subtree. Failed branches leave behind useful information instead of just disappearing.

### Admitting verified improvements

A hypothesis whose evidence beats the current main artifact on held-out evaluation is admitted: its worktree's diff is merged into the main artifact, and the tree updates its root.

---

## Why it matters

- **Cumulative search.** Each attempt's exploration budget compounds across the tree — earlier failures shrink future search space.
- **Bounded context per executor.** Long-lived agents drown in their own context; HTR keeps executor lifetimes short and pushes long-term memory into the tree structure.
- **Safe parallelism.** Worktree isolation means many executors can run in parallel without coordination overhead.
- **Beats flat agents at long horizons.** Arbor + GPT-5.5 hits **86.36% Any-Medal on MLE-Bench Lite**, beating Codex and Claude Code by **>2.5× average relative held-out gain** on six real research tasks under matched task interface and resource budget.

---

## Gotchas & tricks

- **Lessons need to be actionable, not narrative.** The coordinator reads them at expansion time; if they're long retrospectives, the coordinator wastes context. Arbor's lessons are short, mechanistic statements.
- **Search policy matters.** Naive depth-first or breadth-first on the tree leaves obvious improvements on the table. The coordinator's branch-selection policy is a hyperparameter — the paper doesn't disclose theirs in detail.
- **Held-out evaluation is the only thing the coordinator can trust.** Executors can over-fit to whatever local metric they optimize. Always score the candidate artifact on a held-out set before admission.
- **Worktree creation isn't free.** For small experiments the overhead can dominate. Batch micro-experiments inside a single executor when feasible.

---

## Sources

- Paper: *Toward Generalist Autonomous Research via Hypothesis-Tree Refinement* — Hu, Qiu, Dai, Luo, et al. (Renmin U / Microsoft Research), 2026 — [arXiv 2606.11926](https://arxiv.org/abs/2606.11926).
- Concept: Monte Carlo Tree Search — long-line predecessor for tree-structured search policies. See [../post-training/reasoning/mcts.md](../post-training/reasoning/mcts.md).
