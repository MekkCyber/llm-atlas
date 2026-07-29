# Agentic-Search Protocol Distillation
*Depth — closing the gap from proprietary to open-source agentic-search models via a style-normalized intermediate protocol (MAPD).*

**TL;DR:** A distillation recipe for **agentic search** (interleaved reasoning + retrieval) that turns opaque, closed-source teacher trajectories into supervised training data for open-source students, without access to teacher logits. The key move is a **style-normalized protocol** — a canonical structured trace that captures each tool-use decision — used as an intermediate representation between teacher and student.

**Prereqs:** [../agents/README.md](../agents/README.md), [../post-training/rejection-sampling.md](../post-training/rejection-sampling.md)
**Related:** [../post-training/reasoning/long-cot-rl.md](../post-training/reasoning/long-cot-rl.md)

---

## What it is

The gap between proprietary and open-source agentic-search systems is largely a *style* gap in how tool calls and reasoning are interleaved — same task, different trace shape. Outcome-based RL supplies only sparse supervision; token-level distillation needs teacher logits that closed models don't expose. Protocol distillation gets denser supervision from a black-box teacher by projecting trajectories into a shared, structured form.

## How it works

Three stages:

1. **Protocol design.** Define a structured schema for a search trajectory — thought, action (tool call with typed arguments), observation, verification — that is a *canonical* form both teacher and student traces can be projected into.
2. **Teacher trace normalization.** Sample teacher trajectories via API, parse each into the protocol, discarding stylistic variation that doesn't affect outcomes.
3. **Student supervision.** Train the open-source model on the normalized protocol — SFT on the projected trace, or on-policy alignment against protocol-level rewards. Teacher logits are never needed; only the parsed trajectory.

Multi-teacher variants project traces from several proprietary systems into the *same* protocol before mixing, so the student doesn't need to arbitrate stylistic differences.

## Why it matters

Turns closed→open distillation into a well-posed supervised problem: the signal is the *shape of the trace*, not the token-level distribution. Only API access to the teacher is required. Reported gains close much of the agentic-search gap between open-source students and their proprietary teachers on knowledge-intensive multi-step retrieval.

## Gotchas & tricks

- Protocol design is the whole game. Under-specified schemas lose the teacher's decision signal; over-specified schemas fail to parse.
- Trace parsing is brittle when the teacher deviates from expected tool syntax — a lenient parser plus a fallback ("skip this trace") beats a strict one.
- Multi-teacher mixing amplifies bad parses; validate per-teacher parse rate before merging.
- Protocol-level rewards can go stale when the toolset changes; retrain when the search API changes materially.

## Sources

- Paper: *From Proprietary to Open-Source: Bridging the Distribution Gap via Multi-Agent Protocol Distillation in Agentic Search* — Chen et al., 2026 — [arXiv:2607.24280](https://arxiv.org/abs/2607.24280)
