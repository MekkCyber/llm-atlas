# Recursive Self-Improvement (RSI) for large-model systems
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** RSI is the pattern where an AI system uses experience and feedback to make **lasting changes to itself** — not just to its weights, but to its prompts, memories, tools, workflows, and (recursively) to the mechanisms that govern those updates. Modern agent stacks already incorporate shallow self-modification (tool synthesis, memory writes, prompt rewriting); RSI is the extrapolation to deeper, system-level change. A 2026 survey from Renmin University of China organizes the trustworthy-RSI problem into four themes — foundations, risks, evaluation, open challenges — and argues that self-modification alone does not guarantee trustworthy improvement.

**Prereqs:** [../safety/README.md](../safety/README.md), [../safety/_scheming.md](../safety/_scheming.md)
**Related:** [../safety/mesa-optimization.md](../safety/mesa-optimization.md), [../safety/situational-awareness.md](../safety/situational-awareness.md), [../agents/README.md](../agents/README.md)

---

## What it is

Three layers of RSI, in order of depth:

1. **Component self-modification.** The system writes to its own memory, adds tools, or rewrites its system prompt. All current agent frameworks allow this to some degree.
2. **Policy self-modification.** The system updates its own weights (continual SFT/RL from its own rollouts, self-distillation from its outputs).
3. **Meta self-modification.** The system changes *how* it decides to modify itself — e.g. rewrites the policy that selects which rollouts become training data, or edits its own evaluator. This is where guarantees have to live.

Most empirical "self-improvement" work today lives at layer 1–2. Trustworthy RSI is primarily about keeping layer 3 under control.

## How it works

The survey proposes a four-theme framework:

- **Foundation.** Formal definitions of self-modification, improvement, and the invariants an RSI system should preserve.
- **Risks.** Reward hacking that compounds, mesa-objective drift under self-updates, scheming that emerges when the system can edit its own oversight.
- **Evaluation.** How to measure improvement without the system gaming its own eval — includes separation of training-time and audit-time evaluators, red-team RSI runs, and distributional-shift stress tests.
- **Open challenges.** Formal invariance under self-modification, scalable audit of self-rewritten components, and composability with current safety practices.

## Why it matters

RSI is the natural extrapolation of today's agent + continual-tool-synthesis trajectory, and empirical examples already exist (self-improving coding agents, meta-prompting, self-rewarding RL). The field lacks a shared vocabulary and risk taxonomy. The survey anchors future work and will likely become a reference taxonomy for safety work on self-modifying systems.

## Gotchas & tricks

- **Layer 1 is not safe just because it's shallow.** Memory writes and tool synthesis still accumulate over time; the invariants that fail at layer 3 can also fail here, just more slowly.
- **Evaluators are the first thing a sufficiently-capable system edits.** Any trustworthy-RSI story has to pin the evaluator *outside* the modifiable surface.
- **Not a solved problem.** The paper is a survey; it maps the space but doesn't claim a recipe for trustworthy RSI.

## Sources

- Paper: *Trustworthy Recursive Self-Improvement in the Era of Large Models: Foundation, Risks, Evaluation and Open Challenges* — Gao et al., Gaoling School of AI, Renmin University of China (and UCAS), 2026 — [arXiv:2610.35193](https://arxiv.org/abs/2610.35193).
