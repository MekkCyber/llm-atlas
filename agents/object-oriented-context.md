# Object-Oriented Context (CarryOn)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A context runtime for long-horizon agents that structures continuation around **reusable task objects** — work products, sourced findings, application state — rather than compressed interaction history. Context is split into three regions: continuation carriers (the task objects), execution context (goals and constraints), and a live tail of recent turns. "Folding" replaces covered history with its carrier-record representation; only carriers whose supporting history is removed are revised, against evidence and environment checks.

**Prereqs:** _(none yet — agent-folder seeds)_
**Related:** _(none yet)_

---

## What it is

Long-horizon agents accumulate huge transcripts: tool calls, tool outputs, intermediate reasoning. Keeping all of it in context is prohibitively expensive; summarizing it loses provenance; dropping the oldest turns forgets progress.

CarryOn reframes the problem. The agent is not maintaining a transcript — it is maintaining a small set of **task objects** that record the work actually done so far: documents being drafted, sources being gathered, state of the application being driven, outstanding sub-goals. The transcript is a byproduct of producing those objects; once an object is well-formed, the transcript turns that produced it can be discarded.

## How it works

**Three context regions.**
- **Continuation carriers.** Task objects: records of reusable progress (a partial document, a sourced finding, an application state snapshot). Each carrier has structured fields and typed references to its supporting evidence.
- **Execution context.** The persistent goal, constraints, deadlines, and prior commitments.
- **Live tail.** The last $k$ turns in full, so the agent can act on fresh information without a fold cycle.

**Folding.** When context pressure crosses a threshold, the oldest turns in the live tail are **folded**: their contribution to each carrier is distilled, and the raw turns are removed.

**Fold-aware carrier maintenance.** For each carrier:
- Carriers whose supporting history is unaffected by the fold are **carried forward unchanged** (no re-compression).
- Carriers whose supporting history was in the folded range are **revised in place**: the agent re-checks the record against its remaining evidence and any environment affordances (open files, DOM state, running processes). If the record no longer holds, it is updated; if it does, no change.

**Execution context update.** Alongside carrier maintenance, goals and constraints are re-checked for staleness and revised if needed.

## Why it matters

- **Progress survives compression.** Standard history-summary approaches lose typed structure; carrier records keep progress auditable and locally revisable.
- **Lower token cost.** Reported **51.9% and 22.9% total-token reductions** on OSWorld 2.0 and BrowseComp-Plus respectively.
- **Higher task success.** **+5.8% and +7.0%** absolute success on the same benchmarks against both a plain-history baseline and simpler compressors.
- **Generalizes across tool/UX stacks.** Carriers are typed objects defined per workflow, not per model — the same runtime drives GUI agents, research agents, and document-editing agents.

## Gotchas & tricks

- **Carrier schema is a design decision.** For a research agent, carriers include work products and sourced findings; for a GUI agent, application state. A badly-chosen schema loses information at every fold.
- **"Fold" is not "summarize."** Summaries collapse semantics; folds collapse history while preserving typed object references.
- **Environment checks are load-bearing.** Record revision uses the live environment (DOM, filesystem) as a verifier. Without them, revisions can hallucinate.
- **Not a replacement for RAG.** Carriers are the agent's own produced state; external knowledge still flows through retrieval.

## Sources

- Paper: *Pack Light, CarryOn: Building Object-Oriented Context for Long-Horizon Agents* — Lu, McAuley, Zhang, 2026 — [arXiv:2610.18784](https://arxiv.org/abs/2610.18784)
