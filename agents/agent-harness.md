# Agent Harness — Composable, Adaptive Scaffolding
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** An agent's *harness* is everything around the model that mediates how it observes, reasons, and acts — prompts, tools, memory, control flow. The HarnessX paper formalizes harnesses as **typed primitives** combined through a **substitution algebra**, then closes a harness–model evolution loop with **AEGIS**: a trace-driven engine that maps execution traces simultaneously to symbolic harness edits and to RL gradients on the underlying model. Reported +14.5% average across 5 agent benchmarks (up to +44%), with gains concentrated where baselines were weakest.

**Prereqs:** [README.md](README.md)
**Related:** [graph-memory.md](graph-memory.md), [../post-training/_rl.md](../post-training/_rl.md)

---

## What it is

Today's agent harnesses are hand-crafted and static: each new model or task gets a bespoke prompt template, tool list, memory layout, and control flow. Execution traces are abundant during deployment but rarely fed back into the harness design — feedback loops, when they exist, only update the model.

The harness-foundry view splits the system into:

- **Primitives** — typed atomic pieces (a prompt slot, a tool, a memory operation, a router).
- **Substitution algebra** — well-typed rules for replacing one primitive with another or composing them.
- **Evolution engine** — a process that proposes edits from execution feedback.
- **Model** — the LLM that runs inside the assembled harness.

The contribution is treating primitives + algebra as the harness "language" and trace-driven evolution as the harness "compiler."

---

## How it works

### Typed primitives + substitution algebra

Each primitive declares a type: what inputs it consumes (text, tool result, memory key), what it produces (text, tool call, decision). Composition rules check types so a harness configuration is well-formed by construction.

Substitution rules let one primitive be swapped for a richer or alternative one — e.g. replace a single-shot prompt with a retrieval-augmented prompt, or replace a sequential router with a planner-executor pair. Edits propagate through the algebra without breaking type compatibility.

### AEGIS — trace-driven evolution

AEGIS is a multi-agent loop that ingests execution traces and produces *both* harness edits and RL training signals on the same trajectory. Its core idea is an **operational mirror**:

- A symbolic edit (e.g. "swap memory primitive X for Y") corresponds to a parameter-side intervention (e.g. an RL update that biases the policy toward outputs Y expects).
- A failed trace can be addressed either by editing the harness primitive at fault, or by RL-updating the model to behave correctly given the existing primitive.

AEGIS chooses where to apply the fix based on which side is more tractable for the observed failure mode. Successful edits accumulate into the harness; RL updates accumulate into the model.

### Closing the loop

Each evolution cycle:

1. Run the current (harness, model) pair on a task suite, collect traces.
2. AEGIS classifies failure modes; for each, propose a harness edit or an RL update or both.
3. Apply edits + perform RL step.
4. Re-evaluate, keep what improved.

---

## Why it matters

- **Reframes "agent improvement."** Most of the field tunes prompts manually and treats the model as the only learnable component. HarnessX makes the scaffolding itself a learnable component.
- **Gains concentrate where baselines are weakest.** Weak agents benefit most from harness evolution — the harness is the binding constraint, not the model. Strong agents benefit less but still measurably.
- **Composable.** Substitution algebra makes harness reuse across tasks tractable instead of a copy-paste exercise.
- **Co-evolution implies model and harness train together.** Decoupling them (train model, then tune harness) leaves performance on the table because the optimal pair is jointly determined.

---

## Gotchas & tricks

- **Type design is the bottleneck.** Primitive types must be expressive enough to discriminate useful substitutions but rigid enough to prevent ill-formed combinations. The paper presents a starter set; your domain may need extensions.
- **AEGIS attribution is noisy.** Deciding whether a failure is harness-side or model-side is itself an inference problem; expect false attributions, especially early in training.
- **Catastrophic edits.** A harness edit can regress unrelated capabilities. Maintain a held-out regression suite and reject edits that degrade it, even if they improve the target task.
- **RL signal is sparse without good intermediate rewards.** Pair AEGIS with a process reward or with rejection-sampling-style data filtering to keep gradients useful.
- **Watch for primitive bloat.** Substitution is cheap, so the primitive set grows; periodically prune unused primitives to keep the algebra tractable.

---

## Sources

- Paper: *HarnessX: A Composable, Adaptive, and Evolvable Agent Harness Foundry* — 2026 — [arXiv 2606.14249](https://arxiv.org/abs/2606.14249).
- Background: ReAct, AutoGPT, and language-agent frameworks for the manual-harness baseline this generalizes.
