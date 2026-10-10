# Reflective Rulebooks
*Depth — frozen-weight self-improvement via a natural-language rulebook compiled into an executable world model.*

**TL;DR:** Let a *frozen* LLM agent learn explicit world models at inference time by externalising dynamics into a **natural-language rulebook**. The rulebook is then **compiled** into executable code, which generates predictions the agent can verify against new rollouts. Reflection updates the rulebook; recompilation updates the code; verification keeps both honest. A population extension maintains several candidate rulebooks in parallel, closer to evolutionary program synthesis than to standard CoT. Introduced by Memento 3 (2026).

**Prereqs:** [../post-training/_rl.md](../post-training/_rl.md), [../fundamentals/attention.md](../fundamentals/attention.md)
**Related:** [soft-token-memory.md](soft-token-memory.md) · [procedural-world-generation.md](procedural-world-generation.md) · [../safety/cot-monitoring.md](../safety/cot-monitoring.md)

---

## What it is

A memory-plus-program hybrid for frozen LLM agents. Rather than compressing experience into textual rules (which the agent then reads during inference), the rulebook is **compiled into executable code** at every revision. The compiled program predicts observations and state transitions; failed predictions trigger reflection; reflection edits the rulebook text; the next compile tries again.

Weights never move. All adaptation lives in the rulebook ↔ code loop.

## How it works

Four-stage inner loop per episode or batch:

1. **Observation.** The agent collects a trace in the environment.
2. **Reflection.** The frozen LLM reads the trace plus the current rulebook and proposes natural-language revisions (new rules, deletions, qualifications).
3. **Compilation.** The LLM converts the revised rulebook into executable code (typically Python) that acts as a candidate world model: given a state and action, predict the next state and reward.
4. **Verification.** The code is run against held-out traces or new rollouts; mismatches become evidence for the next reflection step.

The **population extension** keeps $N$ rulebooks evolving in parallel; a selection rule (verification score, episode return) decides which ones survive and get recombined — a lightweight program-synthesis search.

## Why it matters

- **Executable predictions you can test.** Unlike CoT or textual memory, a compiled rulebook makes falsifiable predictions, which gives the loop a hard signal that doesn't drift as the text grows.
- **Audit-ready.** The rulebook is human-readable and the code is literally code — oversight and debugging are easier than inspecting latent memory.
- **Strong empirical headline.** Memento 3 clears all 25 public ARC-AGI-3 games with a frozen base model and wins 21:0 across three Atari Pong episodes without weight updates.

A plausible template whenever the environment has learnable deterministic structure (physics, APIs, game rules) that LLMs can describe but not immediately solve.

## Gotchas & tricks

- Compilation quality is the ceiling: a rulebook that doesn't compile or compiles into brittle code can't be verified. Guard with a sandbox and a syntactic retry budget.
- Reflections can oscillate — "add rule, remove rule, add again" — if verification is noisy. Dampen with multi-trace evidence before edits.
- Populations help diversity but multiply compile/verify cost; cap $N$ and expire low-score rulebooks aggressively.
- The approach shines on environments with crisp dynamics; it's weaker in open-ended text domains where no compact rule set exists.

## Sources

- Paper: *Memento 3: Model-Based Recursive Self-Improvement through Reflective Rulebooks* — Zhao et al., UCL / Huawei Noah / U. Liverpool, 2026 — [arXiv:2610.11794](https://arxiv.org/abs/2610.11794).
