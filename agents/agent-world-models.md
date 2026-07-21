# Agent World Models
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** An **agent world model** is a learned predictor of "if I run this action on the current state, what state do I get?" — swapped in for expensive real environment execution during agent RL training or search. The pattern (canonical in Dyna-style RL, revived by DreamerV3) is now landing in *productivity* domains: DSWorld (2026) applies it to autonomous data-science agents, accelerating RL agent training ~14× and search-based inference ~3–6× while remaining competitive on outcome quality.

**Prereqs:** [../post-training/rlvr.md](../post-training/rlvr.md), [../post-training/_rl.md](../post-training/_rl.md)
**Related:** [harness-self-improvement.md](harness-self-improvement.md), [skill-libraries.md](skill-libraries.md)

---

## What it is

Agent training and search burn most of their compute on *actually running* candidate actions in the real environment — running SQL queries, calling APIs, executing code, interacting with a browser. Each candidate action is a real execution.

A world model predicts the environment's response without executing:

$$\hat{s}_{t+1} = f_\phi(s_t, a_t)$$

for some learned $f_\phi$. At training / search time, the model calls the world-model in place of the environment; only occasional real executions are needed to keep the world-model calibrated and to close the loop with reality.

DSWorld's insight: not every action needs simulation. **Cost-aware routing** classifies actions into "cheap enough to run for real" vs. "expensive; simulate instead," giving a hybrid where the world model absorbs only the expensive operations.

## How it works

The DSWorld pipeline (canonical modern instance):

1. **Structured state construction.** Represent the current agent state (data-science workflow) as a structured object — schemas, sample rows, prior operation results — that the world model can consume.
2. **Cost-aware routing.** For each candidate operation, decide: **real execution** (cheap, deterministic, high-fidelity) vs. **world-model simulation** (fast, learned, potentially wrong).
3. **World-model inference.** An LLM-based simulator predicts the next state — the schema after a JOIN, the summary after a groupby, the plot description after a chart call.
4. **Reflective World Model Optimization.** During training, when real-execution results are available, compute prediction-error signals and use them to fine-tune the world model. Error-aware RL keeps the model calibrated on its most-wrong operations.
5. **Downstream use.** RL agent training uses the world model for rollouts; search-based inference uses it for candidate scoring. Real execution is reserved for verification and periodic recalibration.

DSWorld is trained on an 8K-scale transition-trajectory dataset. Reported gains: ~14× faster RL agent training, ~3–6× faster search-based inference, +35.6% over strongest LLM baseline on transition prediction.

## Why it matters

- **Learned simulators are becoming the default way to make agentic RL affordable** — the same move that made game-playing RL possible after DreamerV3.
- **Extends to productivity domains** — DSWorld is the pattern's first solid instantiation in a real productivity domain (data science) rather than games or robotics.
- **Cost-aware hybrid is the practical trick.** Pure world-model training drifts; pure real-execution is unaffordable. A router that spends the real-execution budget where it matters is the point.
- **Complements harness self-improvement.** Better harness × cheaper environment simulation compounds — [harness-self-improvement.md](harness-self-improvement.md) improves the loop, world models cheapen its execution.

## Gotchas & tricks

- **World-model calibration drift.** As the agent policy improves, its action distribution shifts; the world model must re-calibrate or its predictions rot.
- **Router quality is the bottleneck.** Sending too many operations to simulation causes error accumulation; sending too many to real execution defeats the point.
- **Verify with periodic real-execution rollouts.** Even in pure simulation mode, occasionally run the full plan for real to catch simulator drift.
- **LLM-based simulators are cheap per call but brittle** on distributional edges. Combine with a heuristic simulator for operations with structured outputs (e.g. SQL where the schema is known).
- **Not a replacement for real environments in eval.** Report benchmarks with real execution; simulator-only numbers are training-time metrics.

## Sources

- Paper: *DSWorld: A Data Science World Model for Efficient Autonomous Agents* — Yang, Liu, Liu — HKUST (Guangzhou), 2026 — [arXiv:2607.15901](https://arxiv.org/abs/2607.15901).
- Foundational: *DreamerV3* — Hafner et al., 2023 — canonical modern world-model RL.
- Precedent: *Dyna-Q* — Sutton, 1990 — the original world-model-and-RL loop.
