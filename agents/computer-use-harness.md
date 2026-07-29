# Computer-Use Agent Harness (State-First)
*Depth — a code-first, state-grounded harness pattern for long-horizon computer-use agents, popularized by StateAct.*

**TL;DR:** A computer-use agent harness in which the main agent acts on **program state directly** — files, DOM, application backends — through code, and only invokes a screenshot-and-click GUI subagent for the small minority of subgoals that genuinely need vision. A separate *finish gate* verifies the saved artifact structurally, and long tasks are decomposed into fresh subagents to keep the main context focused.

**Prereqs:** [../agents/README.md](../agents/README.md)
**Related:** [../safety/cot-monitoring.md](../safety/cot-monitoring.md)

---

## What it is

Most computer-use agents are perception-first: read a screenshot, decide where to click, repeat. But a screenshot is a lossy rendering of state (files, DOM, app backends), and two very different states can produce identical pixels. A *state-first* harness inverts the default — the main agent inspects and mutates program state via code, and the GUI channel is a fallback for the visual-only subgoals.

## How it works

Three moving parts:

1. **Main agent (code channel).** Operates over program state — reads files, calls APIs, queries the DOM, patches JSON — with no screenshot in its context most of the time.
2. **GUI subagent.** A dedicated screenshot-and-click loop, spawned only when the main agent decides a subgoal is visual-only. In the StateAct evaluation this fires on 28 of 108 tasks and 1.1% of main-agent steps.
3. **Finish gate.** An independent verifier that inspects the saved output structurally (present, saved, correct path, correct schema) and can veto the "done" signal. Grounded in state, not appearance.

Long tasks are decomposed by handing subgoals to fresh subagent contexts so the main agent's context stays focused over hundreds of steps.

## Why it matters

State-grounding shifts the bottleneck from perception to reasoning. Empirically, on **OSWorld 2.0**, StateAct lifts Claude Opus 4.8 from **20.6% → 26.9%** binary success and **54.8% → 61.6%** partial success at **~9× lower cost** than the same model driven by screenshots. A code-only ablation (no GUI subagent) reaches only 45.9% partial, so the hybrid is essential — code-first alone isn't enough.

## Gotchas & tricks

- The pattern needs *reachable* program state. On locked-down desktops or opaque native apps, most subgoals fall back to the GUI subagent and the win shrinks.
- The finish gate is doing real work — without it, the main agent optimistically declares success on writes that never persisted, went to the wrong path, or serialized incorrectly.
- Subagent spawning is a context-management move, not a capability move. Small models used as subagents can still fail hard subgoals; keep the same tier as the main agent.
- Cost gains come from token count, not smart routing — the main agent processes text state rather than tiled screenshots, which dominates the per-step token bill.

## Sources

- Paper: *StateAct: Program State, before Pixels, for Long-Horizon Computer-Use Agents* — Yang et al., 2026 — [arXiv:2607.22798](https://arxiv.org/abs/2607.22798)
