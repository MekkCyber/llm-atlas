# Recursive Harness Refinement (Harness²)
*Depth — treating the agent scaffold as a learnable artifact, improved by the model itself without ground-truth verifiers.*

**TL;DR:** Most agent papers improve the model and freeze the scaffold (prompts, tool schemas, routers, retry logic). **Harness²** does the opposite: freeze the model, improve the scaffold. The refinement runs at two levels: a **task-level** loop that revises *this* task's harness via a verifier-free **Propose–Probe–Compose** cycle, and a **harnessing-level** loop that distills the task-level refinement experience into a meta-harness that continually improves across tasks.

**Prereqs:** [../post-training/rejection-sampling.md](../post-training/rejection-sampling.md)
**Related:** [reflective-agent-training.md](./reflective-agent-training.md) · [progress-reward.md](./progress-reward.md) · [../post-training/rl-prompt-curation.md](../post-training/rl-prompt-curation.md)

---

## What it is

A plug-and-play agent-side training-free improvement procedure. The "harness" is everything *around* the model: system prompt, tool-call grammar, retry/branch policies, memory handling, output post-processors. Harness² learns to edit this scaffold without touching model weights and without requiring external verifiers.

## How it works

**Task-level loop — Propose–Probe–Compose.**

1. **Propose.** Given the current trajectory and current harness, propose `k` candidate harness edits (e.g. "add a sanity-check retry on empty tool responses," "route long queries through the retrieval tool before the reasoning tool").
2. **Probe.** Replay (or re-run on cheap variants of) the task with each proposed harness, collecting *local* traces. No ground-truth answer is needed — the probe measures *contrastive* improvements (does the new harness avoid a failure mode visible in the old trace?).
3. **Compose.** Merge the winning edits into a new harness, carrying forward only the edits whose contrastive probe wins robustly.

**Harnessing-level loop.** Each task's Propose–Probe–Compose produces a (task, harness-edit, outcome) record. These records are themselves distilled by a continual meta-harness that keeps the generalizable edits (format fixes, retry patterns) and discards the task-specific ones. The meta-harness is applied as the default scaffold on the next unseen task.

## Why it matters

- Opens an **orthogonal improvement axis** to post-training. A model fixed in weights can still get smarter in deployment by growing its scaffold.
- **No verifier required** — the Propose–Probe–Compose loop uses contrastive within-trajectory signals, not external labels.
- Reported gains across **three professional benchmarks** and **four model–harness configurations**, including narrowing the gap between a weaker base model and a stronger one at the same compute budget.

## Gotchas & tricks

- **Harness drift.** Over many iterations the harness can grow to include edits that help one task family and hurt another. The harnessing-level loop is supposed to prevent this, but tune the retention threshold aggressively.
- **Probe reliability.** If the probe variant is too cheap, it misses the failure mode the edit is meant to fix. If it's as expensive as the real task, the loop is unaffordable. Pick carefully.
- **Compose conflicts.** Two winning edits can be mutually inconsistent (e.g. one adds a retry on empty, another adds a short-circuit on empty). Define a conflict-resolution policy up front.
- **Scope.** Great for small, modular scaffolds; less clear-cut for monolithic prompts where edits interact unpredictably.

## Sources

- Paper: *Harness²: Recursive Agent Harnessing for an Open World* — Chen, CuiZhu, Dalvi Mishra, Ming, Yu, Han, Lee, Pfister — Google Cloud AI Research / Northwestern, 2026 — arXiv:2610.14224.
