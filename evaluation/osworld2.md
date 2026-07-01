# OSWorld 2.0
*Depth — one specific benchmark, grounded in its source paper(s).*

**TL;DR:** Long-horizon successor to OSWorld: 108 end-to-end computer-use workflows across everyday and professional tasks. Median human time is **~1.6 hours** and a frontier agent averages **~318 tool calls** per task (vs ~30 in v1). Targets challenges absent from v1: streaming interaction, dynamic environments, cross-source reasoning, implicit-state inference, visual-spatial precision. Includes a separate safety audit for safety-sensitive execution.

**Prereqs:** [README.md](./README.md), [../agents/README.md](../agents/README.md)
**Related:** [tua-bench.md](./tua-bench.md), [swe-together.md](./swe-together.md)

---

## What it is

Computer-use benchmarks have been converging on high scores because the tasks are short enough that plan-then-execute frontier agents complete them reliably. OSWorld 2.0 pushes the frontier back out by grounding tasks in *authentic input artifacts* (real docs, real user-profile state) and by demanding hours-long workflows where information arrives mid-task.

## How it works

**Task construction.**
- 108 workflows spanning everyday tasks (email triage across accounts, calendar coordination, receipt filing) and professional tasks (research analysis, software configuration, multi-app data reconciliation).
- Each grounded in a realistic user profile: contacts, past documents, credentials, browsing history. Cross-references between artifacts are the norm, not the exception.
- Median human completion time ~1.6 hours; ~318 tool calls at agent-time using Claude Opus 4.7 max reasoning.

**Challenge phenomena the benchmark targets.**
- Streaming interaction (chat windows, notifications, live updates).
- Dynamic environments (state changes between tool calls).
- Cross-source reasoning (fuse info from multiple apps to answer one prompt).
- Implicit-state inference (facts the user never states aloud).
- Visual-spatial precision (drag-drop, precise clicks).

**Grading.**
- Primary metric: **binary completion at 500 steps**, with a partial-score axis.
- Separate **safety report** auditing safety-sensitive execution paths (e.g. sending emails, spending money).

**Frontier results.**
- Claude Opus 4.8 max thinking + batched tool calls: **20.6% complete / 54.8% partial** — best overall.
- GPT-5.5: ~13%; much more token-efficient.
- Failure modes are *not* GUI control or coding; agents lose track of constraints, miss mid-task information, guess rather than ask, skip verification.

## Why it matters

- **Ends the OSWorld-1 saturation era.** v1 scores had plateaued near ceiling; v2 rebuilds real headroom.
- **Named failure axes drive future training.** The paper's diagnostic breakdown (constraint tracking, verification discipline, information-arrival handling) gives labs concrete gradient signal.
- **Safety report separation.** Auditing safety-sensitive actions independent of task completion is a benchmarking convention worth adopting elsewhere.

## Gotchas & tricks

- **Token cost is significant.** ~318 tool calls per task at the top score; scaling evaluation to many models per release is expensive.
- **Human upper-bound is implicit.** 1.6h median is *median human time*, not accuracy — human success is treated as ~100%.
- **500-step cap can hide "would have finished with more steps" behaviour.** Agents that plateau below the cap fail cleanly; ones that hit the cap mid-progress produce noisier partial scores.

## Sources

- Paper: *OSWorld 2.0: Benchmarking Computer Use Agents on Long-Horizon Real-World Tasks* — XLANG Lab et al., 2026 — [arXiv:2606.29537](https://arxiv.org/abs/2606.29537).
