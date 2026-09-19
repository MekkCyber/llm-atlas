# Agent Harness
*Depth — the scaffolding around a base LLM that turns raw generation into a long-horizon agent.*

**TL;DR:** A coding-agent (or web/GUI-agent) harness is the fixed code path that runs *around* the model: an execution loop plus three orthogonal layers — planning, action space, tool interface, and context management. Recent research (2026) has flipped from framing agents as end-to-end products to studying their harnesses component by component, since a well-designed harness can cut token cost by ~1/3 without touching the underlying model.

**Prereqs:** [../fundamentals/attention.md](../fundamentals/attention.md)
**Related:** [_context-management.md](_context-management.md), [skill-libraries.md](skill-libraries.md), [../evaluation/swe-bench.md](../evaluation/swe-bench.md)

---

## What it is

An **agent harness** is the deterministic wrapper the model runs inside. It defines:

- **Execution loop.** How turns are stitched: single-shot vs. tool-use loop vs. plan-then-act.
- **Planning layer.** Whether the model emits an explicit plan before acting, and how that plan is stored/updated.
- **Action space.** The concrete verbs the model is allowed — a bash-only interface, a fixed toolset with typed schemas, or a mixture.
- **Context management.** How the running context is compacted: rule-based elision, LLM-based summarization, staged rule-then-LLM, or fully recoverable elision.

These are the *only* knobs an operator has once the base model is chosen. Everything else — including reasoning quality — is inherited from the model.

---

## How it works

A harness step runs roughly:

```
observation ─► context manager ─► model.generate(planning?, tools) ─► action
     ▲                                                                    │
     └──────────────── environment.apply(action) ◄──────────────────────┘
```

Each layer can be swapped independently:

- **Planning:** implicit (model plans in latent scratchpad), explicit (model writes a plan file), or scheduled (planner-executor split).
- **Action space:** raw bash / restricted CLI / typed tools / hybrid.
- **Context manager:** truncate-oldest / rule-based-elision (strip stale tool output, keep summaries) / LLM summarize / staged (rule-first, LLM fallback) / recoverable (elided content can be re-fetched).

The empirical finding (Fan et al. 2026, SoL-Pi 2026): staged rule-then-LLM elision dominates cost-efficiency; recoverable elision is dead weight; planning is a cost-saver for strong models and an accuracy-scaffold for weak ones; bash-capable models don't need predefined tools.

---

## Why it matters

Prior agent work reported one number for a whole system (SWE-agent, Devin, OpenHands) so it was impossible to attribute wins. Treating the harness as a set of orthogonal knobs makes ablations legible and lets a good harness ride onto new base models when they land. It's also where the deployment cost is decided — a well-managed context can serve the same task at 1/3 the API cost of a naive one.

---

## Gotchas & tricks

- **Fix the loop before ablating.** Vary planning / action space / context in isolation; if you also change the loop, you can't attribute.
- **Context budget dominates.** Most harness gains show up under a tight context window; at 200k+, the choice of context manager barely matters.
- **Recoverable elision is a trap.** It costs harness code and rarely fires — models don't re-request elided content in practice.
- **Bash vs. typed tools is a model-capability question.** Strong bash models save cost on a bash-only interface; weaker models regress.

---

## Sources

- Paper: *An Empirical Study of Harness Design for Coding Agents* — Fan et al., 2026 — [arXiv:2609.20804](https://arxiv.org/abs/2609.20804).
- Paper: *SoL-Pi: Recursively Scaling Auto-Research Loops for Efficient Agent Harness* — Liu et al., 2026 — [arXiv:2609.20519](https://arxiv.org/abs/2609.20519).
