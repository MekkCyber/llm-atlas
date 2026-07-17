# Harness Handbook
*Depth — a behavior-centric, auto-synthesized index of an agent harness that maps behaviors to their scattered code sites.*

**TL;DR:** Modifying a production agent harness requires locating every code path that implements a target behavior. That behavior is typically spread across files, functions, and execution stages, and existing code search / long-context retrieval leaves the mapping for the developer to recover. **Harness Handbook** builds this mapping automatically via static program analysis plus LLM-assisted behavioral structuring: it organizes implementation knowledge around *behaviors* and links each behavior to the source code that realizes it. **Behavior-Guided Progressive Disclosure (BGPD)** then guides a coding agent from a high-level behavior description down to the concrete edit sites, verifying candidates against current source. On modification requests for two open-source harnesses, Handbook-assisted planning improves localization and edit-plan quality with fewer planner tokens; gains are largest for scattered, rarely-executed, or cross-module code.

**Prereqs:** [agent-harness](agent-harness.md)
**Related:** [failure-attribution](failure-attribution.md), [../evaluation/agent-evaluation-infra.md](../evaluation/agent-evaluation-infra.md)

---

## What it is

Harness Handbook is a **behavior-first representation** of a harness codebase. Standard indexes (symbols, imports, call graphs) organize code by *what it is* (files, functions, modules). Handbook organizes code by *what it does* (behaviors: "route free-form user asks to the QA tool", "compact history when tokens exceed threshold", "retry with reduced tool set on schema error"). Each behavior links back to the concrete code locations that implement it.

The problem it addresses — **behavior localization** — is the bottleneck the paper identifies in harness evolution. A modification request ("stop retrying schema errors more than twice") describes a behavior; the repository is a file tree; behavior can be spread across a planner, executor, retry policy, and formatter. Bridging that gap by hand or with grep is the expensive step.

## How it works

Handbook synthesis is a two-stage pipeline:

1. **Static program analysis** extracts the structural facts — call graphs, control-flow, tool schemas, state transitions, entry points.
2. **LLM-assisted behavioral structuring** clusters these facts into *behaviors* — coherent capabilities described in natural language — and attaches each behavior to the set of code locations it depends on.

The result is a document (a "handbook") organized by behavior, with each entry containing a description, a set of source-code anchors, and the interactions between behaviors.

**Behavior-Guided Progressive Disclosure (BGPD)** is the retrieval protocol on top of the handbook. When a modification request arrives:

1. Match the request to candidate behaviors in the handbook (high-level).
2. Progressively expand to the source-code anchors of the top matches.
3. Verify each candidate site against the current source (to catch drift between handbook and code).
4. Return a ranked set of edit sites plus a suggested edit plan.

BGPD avoids dumping the whole handbook or the whole repo into the planning context — it discloses only the relevant slice, staged by relevance.

## Why it matters

- **Behavior localization is the pain point** in harness evolution — not code generation, not edit synthesis. This paper names it and provides tooling.
- **Fewer planner tokens for better plans.** Handbook-assisted planning uses less context than long-context / retrieval-augmented baselines and produces better edit plans, so it's cheaper *and* better.
- **Wins are largest where naive tools fail** — scattered implementation sites, rarely-executed code paths, and cross-module interactions. Those are exactly the changes that break in production.
- **Harnesses as first-class engineering artifacts.** The paper's framing — that agent harnesses need their own maintenance tooling separate from generic code tools — is a good direction for the field.

## Gotchas & tricks

- **Handbook drift.** The handbook is derived from the source; when source changes and the handbook isn't re-synthesized, retrieval degrades. Rebuild on every merge, or verify candidate sites against live source at query time (BGPD does this).
- **Behaviors that don't cluster cleanly.** Very small utilities or cross-cutting concerns (logging, tracing) resist behavior-first grouping and can be over- or under-attributed. Treat these as a separate "cross-cutting" category rather than forcing them into behaviors.
- **LLM-in-the-loop indexing costs.** Synthesizing the behavior structure requires model calls proportional to codebase complexity — one-time cost amortized across many modification requests, but non-trivial for huge harnesses.
- **Complements rather than replaces long-context retrieval.** For behaviors the handbook missed or bugs unrelated to any known behavior, keep grep and long-context retrieval available as fallbacks.

## Sources

- Paper: *Harness Handbook: Making Evolving Agent Harnesses Readable, Navigable, and Editable* — Wang, Shi, Li, Li, Yu, Yang, Panaganti, Mi, Zhou, Leoweiliang, 2026 — [arXiv 2607.13285](https://arxiv.org/abs/2607.13285). Tencent HY LLM Frontier + partners.
