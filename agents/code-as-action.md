# Code as the Action Interface

*Depth — agent action spaces realized as code executed in a stateful REPL, rather than structured tool calls or single-pass scripts.*

**TL;DR:** Instead of having the agent emit JSON tool calls or write a full Python script up front, give it a **stateful Python kernel** preloaded with task primitives. At each step the agent writes one cell, observes the output (text + values + images), and writes the next cell conditioned on what just happened. The kernel preserves state across cells, so intermediate results become inputs to later decisions. This combines the *composability* of code with the *step-wise observability* of structured tool calls, and is the action interface behind SpatialClaw's +11.2-point gain on spatial reasoning over prior agentic VLMs.

**Prereqs:** [agents/README](README.md)
**Related:** [environment-engineering](environment-engineering.md)

---

## What it is

Three action interfaces dominate practical agents:

1. **Structured tool call** — the model emits `{"tool": ..., "args": ...}`; the harness executes; the result is fed back. Easy to schema-check, but composing operations requires multiple turns and the model can't manipulate intermediate values.
2. **Single-pass code** — the model writes a complete script; the harness runs it; final stdout is the result. Composable, but the model commits to a plan before any observation.
3. **Stateful REPL** — the model writes one cell, the kernel executes and exposes the result (variable bindings, prints, images), and the model writes the next cell with that result already in scope.

"Code as action" usually means option 3. The persistent kernel is the action *space*; primitives loaded into the kernel are the action *vocabulary*.

## How it works

- **Kernel setup**: a Python (or Jupyter-style) kernel is started with input data preloaded and a library of task primitives in the namespace — for SpatialClaw, perception + geometry primitives operating on video frames.
- **Per-step loop**: the VLM/LLM sees a transcript of `[task description, all prior cells, all prior outputs]`. It writes one cell. The kernel executes it (with sandboxing and timeouts). All `stdout`, `stderr`, return values, and rendered images become the next observation.
- **State carryover**: variables defined in cell $i$ remain bound in cell $i{+}1$. The agent can build up intermediate structures (a tracked object, a partial answer, a fitted model) and refine them across steps.
- **Termination**: usually the agent emits a special "answer" call or the harness caps the cell count.

No new training is required — at the limit this is a training-free pattern that runs against an existing strong code-capable VLM/LLM.

## Why it matters

- **Composition without commitment.** Step-wise observation lets the agent branch on intermediate results, which is exactly what open-ended 3D/4D reasoning, data analysis, and debugging require.
- **Bridges the VLM-tools gap.** A VLM can render an intermediate frame, look at it, and decide what to compute next — something structured tool calls can't express cleanly.
- **Pattern transfer**: the SpatialClaw paper reports +11.2 points on spatial reasoning vs. the prior spatial agent, *with no training-specific adaptation* — i.e. the action-interface design alone moves the needle. The same pattern is informally validated by Codex- / GPT-5-class general-purpose code agents.

## Gotchas & tricks

- **Sandbox carefully.** A stateful kernel that can do anything Python can do is a strong primitive — and a strong attack surface. Run inside a container with cap-dropped privileges, no network egress unless the task requires it, and aggressive timeouts.
- **Watch the context length.** Every cell's output goes into the next prompt; long-running tasks need a summarization or pruning policy.
- **Primitive design is the leverage.** The kernel's preloaded library *is* the action vocabulary. Generic Python without task primitives forces the model to derive everything from scratch.
- **Reward / supervision is harder than for structured tools.** There's no schema to check per step, so trajectory-aware judging is needed for training or eval (see trajectory-aware judging in the WeaveBench discussion).

## Sources

- Paper: SpatialClaw — Hachiuma et al. (2026) — [arXiv:2606.13673](https://arxiv.org/abs/2606.13673)
- Background: code-as-action has informally been the dominant interface for SWE-bench / Codex-style agents since ~2024; SpatialClaw formalizes it for the visual reasoning setting.
