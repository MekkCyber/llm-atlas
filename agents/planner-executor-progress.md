# Planner–Executor with Progress Head
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A two-model agent scaffold that decomposes long-horizon tasks into (a) a **planner** that reads long history and emits a compact segment-level plan + visual guidance, and (b) a **fixed-context executor** that runs the plan and predicts a **progress scalar** used to trigger replanning. Keeps executor latency roughly constant as task history grows. Introduced by **MaP-WAM** (2026) in a robotics VLA setting, but the recipe is directly applicable to LLM agents.

**Prereqs:** [README.md](./README.md)
**Related:** none (agents folder is currently sparse)

---

## What it is

Every serious agent scaffold hits the same wall: as the trajectory grows, either the executor's context grows (making inference cost scale with history) or the summarizer discards visual/structural detail (breaking downstream steps). React → Reflexion → Voyager all handle this with ad-hoc summaries; MaP-WAM formalizes the pattern.

Two models, one loop:

- **Planner (long-context, cheap-to-call)** — reads the full multimodal episodic memory (completed segment records: language instructions + sparse visual context) and emits the *next segment* as a compact plan: a language description + visual guidance.
- **Executor (fixed-context, expensive-to-call)** — conditions only on the current plan and predicts (a) action chunks and (b) a **progress scalar** in [0, 1] measuring how far into the plan it is.

A plan-observation alignment module triggers the planner again when progress saturates or drifts.

## How it works

Per step:

1. Planner input: full episodic memory (segments so far) + current observation → emits next language plan + sparse visual guidance patches.
2. Executor input: current plan + current observation → action chunk $a_t$ + progress $p_t$.
3. Alignment check: if $p_t$ crosses a threshold (plan complete) or observation-plan divergence exceeds tolerance, invoke the planner again with the updated episodic memory.
4. KV cache is reused across steps within a plan; structured attention makes both planner and executor cache-friendly.

Because the executor's context is bounded (current plan, not full history), its per-step latency is roughly constant regardless of how long the task has been running.

## Why it matters

- **Constant-time execution.** In LLM agent settings, executor latency growing linearly with history has been the practical bottleneck; the two-model split with a bounded executor context fixes this cleanly.
- **Progress head is the missing signal.** Existing agent scaffolds rely on the planner's opinion (or a timeout) to decide when to replan; a learned per-step progress scalar makes replanning event-driven rather than heuristic. Applies to browser agents, coding agents, long-form writing.
- **KV-cache reuse falls out.** Both models have fixed inputs within a segment; standard KV caching gives large speedups on repeated calls with only the observation changing.

## Gotchas & tricks

- **Progress head is easy to overfit.** In supervised settings the head can memorize timing; the paper trains it with plan-observation alignment as auxiliary supervision so it stays grounded in observed state.
- **Plan compactness matters.** If the planner emits a plan too long to fit the executor's budget, the whole point is lost. Segment-level granularity (a few dozen tokens + a handful of visual patches) is the sweet spot.
- **Replan cost.** Each replanner call is expensive because the planner reads long history; setting the replan threshold too aggressively wipes out the fixed-context savings. Threshold + hysteresis tuning is a first-class hyperparameter.
- **Not just for robotics.** The paper's setting is manipulation, but the pattern (planner over long history, executor over fixed plan window, progress-triggered replanning) is exactly what LLM agents on browser / desktop / long-form writing tasks need.
- **Different from React.** React interleaves thought and action tokens in one model; planner-executor-progress puts them in *different* models with different context budgets. The separation is what enables the fixed-length executor.

## Sources

- Paper: *Memory as Plans: World-Action Modeling with Memory-Grounded Planning* — Xie, Zhao, Zhang, Wang, Wang, Liu, Zhang (NTU / HIT / Shandong), 2026 — arXiv:2609.11561.
