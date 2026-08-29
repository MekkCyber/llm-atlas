# PILOT — Supervisor-Worker Live Self-Improvement
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Standard agent self-improvement harvests trajectories *after* execution ends — it cannot redirect an active run, and lessons from run N only apply from run N+1 onward. **PILOT** is a supervisor-worker harness where a separate supervisor process runs concurrent with the worker and can (1) **live-steer** — redirect or abort the active worker mid-run — and (2) **live-evolve** — distill procedures and failure modes emerging during the current run into reusable skills and memory that become available immediately. Ranks first in 5/6 configurations across Terminal-Bench 2.0 and self-improvement benchmarks with GLM-5.1 and Kimi-K2.6.

**Prereqs:** [README.md](README.md)
**Related:** [skill-evolution.md](skill-evolution.md), [../post-training/reasoning/long-cot-rl.md](../post-training/reasoning/long-cot-rl.md)

---

## What it is

Single-agent self-correction mixes execution with self-assessment inside one context: the worker has to notice its own failure mid-response. This works for short tasks but breaks over long horizons because the context gets crowded and the worker's judgment degrades along with its work. Subagent-delegation approaches separate execution from planning but usually can't interrupt an active subagent once it's launched.

PILOT splits the two roles into two processes. The worker executes. The supervisor watches the worker's stream and either lets it run, steers it (injects a course-correction), aborts it, or captures a distilled skill/memory update — while the worker is still going.

## How it works

**Two coupled mechanisms:**

- **Live steering.** A separate supervisor tails the worker's action stream. Cheap rule-based triggers (repeated errors, loop detection, budget overrun) escalate to the supervisor, which decides between three actions: `continue`, `steer(instruction)` (inject a message into the worker's next observation), or `abort_and_restart(new_plan)`. The worker's context is preserved when steered; only new steering messages are appended.
- **Live self-evolution.** As the run unfolds, the supervisor extracts (procedure, precondition, outcome) triples from segments the worker got right, and (failure mode, symptom, correction) triples from segments it got wrong. Both are written to a shared skill/memory store that the worker reads from on subsequent tool calls in the same run.

Both mechanisms share the same LLM backbone (frozen); the split is procedural, not architectural.

## Why it matters

- **On-policy self-improvement.** Lessons apply to the current run, not the next one. On long-horizon tasks this is the difference between salvaging a run and losing it.
- **Token efficiency.** Mean output tokens fall **42.9–47.4%** vs baseline harnesses because steered abort-and-restart is much cheaper than letting a worker chase a dead branch to the token limit. Successful-evals-per-M-tokens rise **110–134%**.
- **Model-agnostic.** Same harness improves both GLM-5.1 (+14.6 pts) and Kimi-K2.6 (+12.4 pts) on the paper's self-improvement setting.
- **First-in-class on Terminal-Bench 2.0.** Up to +9.8 pp over counterpart harnesses.

## Gotchas & tricks

- **Supervisor cost is real.** A second LLM watching every worker step doubles per-step cost — the paper amortizes with cheap rule-based triggers so the supervisor's LLM is queried only on escalation.
- **Steering must not shred context.** Injecting steering messages mid-conversation can confuse the worker; the paper wraps them in a distinguishable role tag.
- **Skill/memory writes must be atomic.** A partial write during a live run can leave the store inconsistent for subsequent reads within the same run.
- **Not a substitute for training.** PILOT is a *harness*, not a fine-tune — the gains stack with model improvements but don't replace them.

## Sources

- Paper: *PILOT in the Loop: Live Self-Improvement for Long-Horizon Agents* — Xiao et al. (Xiaohongshu / PolyU), 2026 — arXiv:2608.26530.
