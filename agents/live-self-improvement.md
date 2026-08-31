# Live self-improvement for long-horizon agents
*Depth — supervisor-worker harness (PILOT) that redirects and distills mid-run.*

**TL;DR:** Standard agent self-improvement is post-hoc: analyze the trace *after* the run ends and update memory / skills for next time. Long agent runs (browser tasks, coding, computer use) can burn many hours before ending, so post-hoc improvement is often too late. PILOT introduces two coupled in-run loops: a supervisor process that can steer or abort the active worker, and a distillation pipeline that compiles emerging procedures into skills the same run can use.

**Prereqs:** [README.md](README.md)
**Related:** [skill-evolution.md](skill-evolution.md), [skill-retrieval.md](skill-retrieval.md)

---

## What it is

A harness pattern for long-horizon agent execution. Two processes run concurrently:

- **Worker** — the primary agent doing the task.
- **Supervisor** — a separate LLM process watching the worker's trace with authority to intervene.

The supervisor is not a critic model post-hoc; it acts in the same tick as the worker.

## How it works

**Live steering.** The supervisor reads the worker's rolling trace, tool calls, and intermediate outputs. When it detects a doomed trajectory (repeated errors, off-task drift, budget exhaustion pattern), it can:

- Inject a corrective instruction.
- Abort a subtask and prescribe a re-plan.
- Force a tool call (e.g., "read the file before editing").

**Live self-evolution.** When the supervisor observes a successful procedure that the worker discovered — or a failure mode not yet in the skill library — it distills it into a reusable skill or memory entry immediately. The same run can then use the new skill before the run ends.

Both loops share the same trace as ground truth; the supervisor's outputs are audited and can be reverted (steering) or rolled back (distillation) if downstream reward drops.

## Why it matters

For runs measured in hours (day-long agent workflows, browser sessions, multi-file coding), waiting until the end to learn is prohibitive — the runs are expensive, and by the time the trace is analyzed the operator has moved on. Live steering makes hour-long agent runs *saveable*: catch the drift at minute 10, not at post-mortem. Live self-evolution collapses the "learn once per run" ceiling that has capped skill-library growth in prior work.

## Gotchas & tricks

- Supervisor cadence — polling every tool call is expensive; a heartbeat every N steps + explicit "supervisor requested" hook works better in practice.
- Steering must be reversible; log every intervention so bad supervisor calls can be undone.
- Skill distillation mid-run risks poisoning the library with a not-yet-validated skill; keep new skills in a probationary pool until the run ends and the reward is known.
- Doesn't remove the need for post-hoc analysis — it complements it by shortening the *within-run* feedback loop.

## Sources

- Paper: *PILOT in the Loop: Live Self-Improvement for Long-Horizon Agents* — Xiao, Sun, Wu, Hu, Li, Jiang et al., 2026 — [arXiv:2608.26530](https://arxiv.org/abs/2608.26530)
