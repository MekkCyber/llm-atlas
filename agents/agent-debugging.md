# Agent debugging (Detect–Attribute–Recover–Rerun)
*Depth — a closed-loop debugging methodology for LLM-agent trajectories.*

**TL;DR:** LLM-agent failures are hard to debug because the step where an error *surfaces* is rarely the one that *caused* it. AgentDebugX formalizes debugging as a four-stage closed loop — Detect → Attribute → Recover → Rerun — with a multi-turn root-cause-analysis module (DeepDebug) at its core. On the Who&When benchmark, DeepDebug reaches 28.8% exact agent-and-step attribution accuracy on qwen3.5-9b (vs 21.7% single-pass baseline); on GAIA, it repairs 13/73 failed tasks in one rerun (vs 4–6 for self-correction), lifting overall accuracy from 55.8% to 63.6%.

**Prereqs:** *(none in the current graph)*
**Related:** [../agents/README.md](../agents/README.md), [../evaluation/README.md](../evaluation/README.md)

---

## What it is

Most agent post-mortems look at the failing step in isolation. That misses **causal attribution**: the actual bug is often several steps earlier, hidden inside an implicit assumption or a bad tool result. Existing observability tools replay traces but don't diagnose. Self-correction tools patch symptoms without root-cause reasoning.

Agent debugging as a discipline treats each failed trajectory as a *diagnostic problem*: identify which agent, which step, and (crucially) *why* — then translate the diagnosis into a targeted recovery that a rerun can validate.

## How it works

The DAR-R loop:

1. **Detect.** Flag failure — a stopping condition, a bad tool response, an assertion, or a downstream check.
2. **Attribute.** Multi-turn root-cause diagnosis over the *whole* trajectory. DeepDebug combines (i) global trajectory understanding — read the run as a graph, not a sequence; (ii) structure-guided investigation — target agents, steps, and message roles; (iii) cross-examination — generate rival hypotheses and interrogate them against the trace.
3. **Recover.** Translate the attributed cause into a targeted repair: a rewritten instruction, a fixed tool call, a corrected assumption.
4. **Rerun.** Execute the fix and check whether the failure is gone. If not, restart the loop with the new trajectory as evidence.

Optional **Error Hub**: scrubbed failure–diagnosis–repair bundles are shared and reused as debugging memory, so subsequent agents recognize known failure modes on sight.

## Why it matters

Agent evaluations have mostly measured pass/fail on curated tasks. This is the first framework that measures the debugging capability itself as a first-class quantity, with an attribution accuracy metric on Who&When and a real-world repair delta on GAIA. It's also the first LLM-agent equivalent of the "stack trace + post-mortem + shared knowledge base" loop that carried software engineering for decades.

## Gotchas & tricks

- Attribution accuracy is a hard metric: exact-agent-and-step matching penalizes near-misses. Rank-based metrics tell a different story.
- Cross-examination between rival hypotheses is where multi-turn beats single-pass — a single-shot judge tends to anchor on the surface failure.
- Error Hub scrubbing (removing PII, secrets, and task-specific data before sharing) is a non-trivial engineering problem the paper flags but doesn't solve.

## Sources

- Paper: *AgentDebugX: An Open-Source Toolkit for Failure Observability, Attribution, and Recovery in LLM Agents* — Zhu et al., 2026 — [arXiv:2607.18754](https://arxiv.org/abs/2607.18754)
- Repo: open-sourced with a Python library, CLI, web console, and installable skill (see paper).
