# Stochastic Reflective Memory Ascent (SRMA)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** In multi-agent LLM systems where an orchestrator decomposes tasks for workers and iterates via textual reflection, "free-form" text-only reflection cannot uniformly improve across environments — an impossibility result. **SRMA** replaces the default LLM-judge reflection loop with a rule: *accept a memory update only when a grounded (task-specific) evaluator certifies a strict reduction in expected risk.* Under calibration assumptions this yields geometric or polynomial convergence to the exact optimum.

**Prereqs:** [README.md](./README.md), [../post-training/rlvr.md](../post-training/rlvr.md)
**Related:** [../post-training/reasoning/orm.md](../post-training/reasoning/orm.md), [../post-training/_rl.md](../post-training/_rl.md)

---

## What it is

Reflection loops in multi-agent LLM systems typically look like: worker fails → orchestrator (or a judge LLM) writes down a textual lesson → the lesson is appended to a memory that guides future rollouts. The failure mode: a text-only judge can rubber-stamp lessons that do not actually reduce failure probability, and repeated reflection can drive the system in circles.

SRMA formalizes the loop as a stochastic Markov chain over semantic memory states and gates each proposed memory update through a grounded evaluator (one that measures a real task-relevant risk — verifier pass rate, reward-model score, held-out win-rate). Only updates the evaluator certifies as strict improvements are committed.

## How it works

Cast the system as a bilevel coordination game:

- **Upper level.** Orchestrator picks a task decomposition; its quality controls *worker equilibrium slack* — the room workers have to fail on their assigned subtasks without breaking the whole.
- **Lower level.** Workers execute; failures produce candidate reflections.
- **Memory chain.** Reflections propose transitions in a semantic memory-state space $\mathcal{M}$. Free-form reflection = arbitrary transitions.

SRMA replaces the free-form transition with:

```
propose Δm from failure trace
if grounded_evaluator(m ⊕ Δm) < grounded_evaluator(m) - τ:
    commit m ← m ⊕ Δm
else:
    discard
```

`grounded_evaluator` returns expected risk (lower = better) under the task's real reward. `τ > 0` is the strict-improvement margin; a positive margin is what excludes zero-benefit random walks in memory space.

The paper proves: (a) with any text-only evaluation gate, there exist environments where the reflection loop cannot uniformly improve; (b) with SRMA and a calibrated grounded evaluator, the chain converges to the risk-minimizing memory at geometric rate (fast) or polynomial rate (slow but reliable) depending on how tight the evaluator is.

## Why it matters

- **Names why "just reflect harder" often fails.** Text-only reflection loops are provably weaker than grounded ones, not just empirically. Fixes intuition, not just numbers.
- **Bridges RLVR into agent design.** The "grounded evaluator" is the same object as an RLVR verifier — SRMA lifts verifiable-reward machinery from post-training up to inference-time reflection.
- **Verified on three qualitatively different tasks.** Resource-contest game (game-theoretic), Overcooked (coordination), SWE-bench (code repair) — showing the mechanism holds beyond one domain.

## Gotchas & tricks

- **Evaluator calibration is the whole game.** An uncalibrated evaluator (over-confident, cheatable) breaks the convergence guarantee — SRMA can still admit bad updates. Prefer verifiers with explicit ground truth over learned judges when possible.
- **Strict-improvement margin τ.** Too small and noise-driven updates leak in; too large and the chain stalls in local optima. Cross-validate.
- **Not a substitute for task decomposition quality.** SRMA fixes the reflection step, not the orchestrator's decomposition. Poor decompositions produce large worker equilibrium slack and SRMA is slow to compensate.
- **Memory-state definition matters.** SRMA's Markov analysis assumes a well-defined memory-state space; ill-defined "just prepend text" memories violate the analysis's independence conditions.

## Sources

- Paper: *Bilevel Coordinated Reflection: A Game-Theoretic Approach to Multi-Agent LLM Systems* — Yuxiang Chen, Yuxuan Huang, Meng Fang, Weilin Luo, Jun Wang — UCL / Huawei — arXiv:2609.02750 (2026).
