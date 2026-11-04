# Milestone-Guided Search
*Depth — recursive self-improvement for long-horizon agents via checkpointed verified intermediate states.*

**TL;DR:** Long-horizon agent tasks fail at specific subgoals; naive retry restarts from scratch and burns compute on the parts the agent already solved. **Milestone-guided search** (NorthCUA, 2026) saves verified intermediate states — environment snapshot + agent context + working-memory notes — as **milestones**. A retry attempt branches from the latest milestone rather than from zero, so later attempts only have to solve what the previous attempt couldn't. Combined with a skill library (reusable verified procedures), this makes previously-unsolvable tasks reachable through compounding partial progress.

**Prereqs:** [experience-flywheel.md](./experience-flywheel.md)
**Related:** [../post-training/_rl.md](../post-training/_rl.md) · [../post-training/rejection-sampling.md](../post-training/rejection-sampling.md)

---

## What it is

A **milestone** is a verified intermediate state on the way to task completion. Operationally it bundles:
- The **environment snapshot** (OS/VM state, file system, browser state, open applications) at that point.
- The **agent context** (its recent screenshots-with-overlays, working-memory notes, planned next steps).
- The **verifier witness** — evidence that the intermediate goal was actually reached (test passed, file exists, page loaded).

A **search** over milestones means: when task `T` fails, don't start the next attempt at `T`'s initial state. Instead start it at the latest milestone the agent did reach, and explore different continuations from there.

## How it works

1. **Attempt** task `T`. As the agent works, its judge/verifier emits partial-success signals: "sub-goal 1 of 3 is done," "file was saved correctly but formatting is wrong."
2. **Checkpoint.** At each partial success, snapshot the environment + agent context. Tag it with the sub-goal it satisfies. These snapshots become milestones.
3. **Fail.** The task doesn't complete on this attempt.
4. **Retry from milestone.** For the next attempt on `T`, load the latest milestone. The agent sees the context and environment as if it had just reached that point naturally. It only has to solve from here forward.
5. **Branch at milestones.** Multiple retries can branch from the same milestone with different sampling, adding breadth to the search over the remaining subtask.
6. **Skill extraction.** If a milestone is re-reached reliably on many tasks, it (and the actions that lead to it) are promoted to a **skill** — a reusable procedure in the skill library that future tasks can invoke.

## Why it matters

- **Compounding progress.** Each failed attempt still leaves verifiable gains (new milestones). Tasks the model can't solve in one attempt become solvable across many.
- **Compute amortization.** Retries skip the already-solved prefix; the cost of a retry scales with the remaining work, not with total task length.
- **Natural exploration bias.** The search space gets explored breadth-first from the hardest unsolved frontier, not depth-first from initial state — the right place to explore in a long-horizon task.
- **Produces skill catalog.** Over many tasks, the recurring milestones become a library of reusable procedures (skill-guided exploration), accelerating future tasks' early phases.

## Gotchas & tricks

- **Snapshot reliability.** VM / browser snapshotting is non-trivial; snapshots must be deterministic (same snapshot → same observed state) or retries see different environments than the milestone promised.
- **Verifier granularity.** Needs intermediate verifiers for subgoals, not just end-state verifiers. The Experience Flywheel's verifier design should contemplate milestones.
- **Milestone bloat.** Not every intermediate state is worth keeping; store milestones only where a verifier confirms meaningful progress, or retrieval over milestones becomes the bottleneck.
- **Distribution shift at retry time.** The agent is restored to a milestone it didn't create itself; its own policy might not emit from there. Policies trained on milestone-starting trajectories handle this best.
- **Task rework on model update.** Milestones taken under old model's screen-reading behavior may not round-trip for a new model. Retake milestones when the base model changes.

## Sources

- Paper: *NorthCUA Technical Report* — NorthCUA Team, Shanghai Qiji Zhifeng, 2026 — [arXiv:2611.02102](https://arxiv.org/abs/2611.02102).
- Parallel work: *CUA-Master* ([arXiv:2611.03126](https://arxiv.org/abs/2611.03126)) — snapshot-rollback retry in a nested double-loop framework.
