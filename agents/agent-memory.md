# Hierarchical Agent Memory (Workflow / Subtask / Function)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Distill a strong teacher agent's successful trajectories into **three memory layers** — Workflow (task-level strategy), Subtask (mid-granularity behavioral examples), and Function (per-tool calling conventions and pitfalls). Inject Workflow + Subtask *proactively* at task start; retrieve Function memory *reactively* on tool-calling errors. A 4B–8B student agent using teacher-distilled hierarchical memory beats prior memory baselines with **zero training**: +27.2 pp on AppWorld, +11.2 pp on BFCL V3, +3.4 pp on ToolSandbox.

**Prereqs:** [README.md](README.md), [../post-training/on-policy-distillation.md](../post-training/on-policy-distillation.md), [../post-training/rejection-sampling.md](../post-training/rejection-sampling.md)
**Related:** [self-developing-harness.md](self-developing-harness.md)

---

## What it is

"Agent memory" is a catch-all term for anything an agent stores across steps to inform future decisions. Prior work either shoved everything into a single retrieval index (episodic) or hand-crafted small skill libraries (procedural). Hierarchical Agent Memory splits by *granularity* and *retrieval timing*, matching the level of abstraction to the moment it's needed.

Three memory types, mined from teacher trajectories:

- **Workflow memory.** Task-level strategies — the shape of the plan for a class of task. Coarse, few entries per task family.
- **Subtask memory.** Concrete behavioral examples at intermediate granularity — how a specific subtask (e.g. "authenticate to service X") was executed in a successful trajectory.
- **Function memory.** Per-function calling conventions, common argument mistakes, and error-recovery patterns for individual tool calls.

## How it works

**Distillation.** Run a strong teacher agent (e.g. GPT-5-mini) on the target task suite. Extract successful trajectories. Cluster and abstract them into the three memory layers offline. This is training-free from the student's perspective — the student's weights are never updated.

**Runtime.** For a new task:
1. **Proactive injection.** At task start, inject the top-matched Workflow and Subtask memories into the student's context. This front-loads the agent with the teacher's plan and behavioral templates.
2. **Reactive retrieval.** If a tool call fails or returns an unexpected type, look up the Function memory for that specific tool and inject it. Errors are the trigger — most Function memory is never touched.

The split matters: injecting all Function memory proactively drowns the model in details it doesn't need; withholding Workflow/Subtask until failure means the student has to learn strategy from scratch. Matching level to moment is the whole trick.

## Why it matters

- **Distillation without weight updates.** Small agents catch up to frontier agents on tool-use benchmarks purely through better memory extraction. Cheaper to iterate on than weight-level distillation, and safer for deployed models.
- **Concrete numbers.** +27.2 pp on AppWorld (a hard multi-app agent benchmark), +11.2 pp on BFCL V3 (function-calling accuracy), +3.4 pp on ToolSandbox — with 4B–8B students against a GPT-5-mini teacher.
- **Orthogonal axis.** Weight-level distillation, RL post-training, and scaffolding evolution (Ouroboros-style) all target different capabilities. Hierarchical memory attacks the "you can teach an agent to use tools by showing it, not by retraining it" angle.
- **Fits the small-agent economics.** For on-device or budget-constrained agents where post-training is impractical, memory extraction is the highest-leverage improvement path.

## Gotchas & tricks

- **Memory quality is teacher-dependent.** Bad teacher trajectories → bad memory → student learns the teacher's mistakes. Filter aggressively before abstracting.
- **Retrieval quality caps the benefit.** If the router picks the wrong Workflow memory for a novel task, the student is worse off than with no memory. Coarse-grained similarity search is usually sufficient at small `N`; embedding + rerank at scale.
- **Function memory storage grows with tool surface.** For agents with hundreds of tools, Function memory becomes its own retrieval problem — one per-tool document is manageable; per-tool-per-error-mode explodes.
- **Not a substitute for real world model.** Memory helps replay teacher patterns; it doesn't teach the agent *why* a step was correct. Novel tasks with no matching Workflow memory still bottleneck on base-model capability.
- **Timing choices are load-bearing.** Injecting Function memory proactively looks tempting but consistently hurts — the student spends context on details irrelevant to the current step. Keep the reactive/proactive split.

## Sources

- Paper: *Agent Memory Distillation: Empowering Small LLM Agents with Hierarchical Teacher Memory* — Kim, Kim, Hwang, 2026 — KAIST / DeepAuto.ai.
- Related: [../post-training/rejection-sampling.md](../post-training/rejection-sampling.md) — mining successful teacher trajectories is the supervision analogue of rejection sampling.
