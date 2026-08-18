# Parallel Reasoning (Reasoning-Idle-Window Filling)
*Depth — decode auxiliary reasoning branches during the Action-Observation wait in a ReAct-style agent.*

**TL;DR:** In ReAct-family agent loops, the phase between issuing an action and receiving an observation is dead wall-clock — the model is waiting on the environment. Parallel-reasoning frameworks (Second Thought is the canonical instance) *fork* several auxiliary reasoning branches the instant each Thought phase ends, decode them concurrently with the Action-Observation wait, and merge their outputs back when the observation arrives. Training-free, model-agnostic, and unlike serial extra reasoning it doesn't inflate main-thread decoding.

**Prereqs:** none (works with any ReAct-style loop and any reasoning-capable LLM).
**Related:** [../post-training/reasoning/long-cot-rl.md](../post-training/reasoning/long-cot-rl.md)

---

## What it is

A serving-side inference trick for agent loops. A ReAct step is `(Thought → Action → Observation)`. The Action-Observation gap can be tens to thousands of ms (tool call, API roundtrip, tool execution). During that gap, the model isn't doing anything useful. Parallel-reasoning schemes decode extra thoughts in that window so the *next* Thought starts with more context — without lengthening the main thread's sequential decoding.

## How it works

Per Second Thought's recipe:

1. The main loop finishes a Thought and issues an Action.
2. At the same instant, spawn `N = 4` auxiliary decoding branches (Second Thought's tested number). Each branch's prompt is the current context plus a diverse "second-thought" seed (e.g., "reconsider assumptions", "consider counterexample", "plan for observation X").
3. Branches decode concurrently with the tool call. All decoding shares the model server (same batch on the same GPU when possible).
4. When the observation returns, terminate any still-running branches and merge their outputs into the context (concatenation with headers, or summarization).
5. The next Thought conditions on `context + observation + merged auxiliary thoughts`.

Nothing about the model or training changes. All the work is on the inference orchestrator.

## Why it matters

- **Free wall-clock win.** Second Thought reports up to 43% (avg ~20%) reduction in main-thread decoding across 6/9 benchmark×model pairs, with Pass@1 unchanged or improved.
- **Compute-matched dominance.** A control that gives the *main thread* the same extra token budget is strictly worse — the win comes from parallelism, not from more thinking tokens.
- **Composes with speculative decoding, tool caching, etc.** — same layer of the stack (orchestration), different lever (idle-window utilization).

## Gotchas & tricks

- Merging strategy matters more than branch count. Naive concat can drown the observation; summarization or filtering is often needed.
- Auxiliary-branch prompts need to be diverse — four branches asking the same question is wasted parallelism.
- GPU saturation: if the model server is already near capacity, the parallel branches steal time from other requests. Best gain is at low-to-moderate load.
- Not helpful for zero-latency tools (no wait window) or reasoning tasks that don't call tools.

## Sources

- Second Thought: Reasoning in Parallel as LLM Agents Act and Observe — Zhensu Sun et al., 2026 — [arXiv:2608.13667](https://arxiv.org/abs/2608.13667)
