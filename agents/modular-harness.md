# Modular Harness
*Depth — decomposing an agent scaffold into five independently evolvable modules.*

**TL;DR:** A monolithic agent harness entangles unrelated mechanisms, making attribution of failures and safe evolution hard. ModularRSI (Wu et al., 2026) proposes splitting the harness into five functional modules — Agent Loop, Tool Use, Observation Management, Context Management, and Task Completion Detection — each evolved independently within a restricted modification scope, followed by an integration stage that resolves conflicts between the evolved modules.

**Prereqs:** [harness-evolution](harness-evolution.md)
**Related:** [_recursive-self-improvement](_recursive-self-improvement.md), [native-skill-routing](native-skill-routing.md)

---

## What it is

A structural principle for agent harnesses used in coding / terminal / SWE-style long-horizon tasks. The harness is decomposed into five modules with **explicit interfaces**, and each is edited in isolation during harness evolution. An integration stage then combines them into a unified harness and reconciles inconsistencies. Introduced by ModularRSI as the missing complement to contrastive-trajectory attribution.

## How it works

The five modules and their responsibilities:

1. **Agent Loop.** The top-level control flow: think → act → observe → repeat. Owns turn ordering and termination hooks.
2. **Tool Use.** The set of tools, their schemas, and the argument-construction policy.
3. **Observation Management.** How raw tool output is normalized, summarized, or filtered before entering context.
4. **Context Management.** How past observations, thoughts, and tool calls are retained, evicted, or compressed as the trajectory grows.
5. **Task Completion Detection.** How the agent decides it's done — including safeguards against premature exit or infinite loops.

Evolution proceeds per module. Contrastive attribution (successful vs. failed trajectories on the same task) surfaces recurring deficiencies per module. Each module's proposal is validated against a held-out subset before acceptance. The integration stage runs the combined harness against the whole validation set and resolves conflicts (e.g. an Observation Management edit that assumes an old Context Management policy).

## Why it matters

Monolithic harness evolution has two failure modes: (a) attribution is fuzzy because one prompt change touches everything, and (b) unrelated modules co-evolve accidentally, producing brittle recipes that don't transfer. The modular split gives clean attribution *and* a natural evolution scope. Empirically, evolved harnesses transfer across foundation models — the rare property that makes harness evolution worth doing at all.

## Gotchas & tricks

- **Interface drift.** Modules must respect stable interfaces or the integration stage becomes an integration-hell stage. Define the module contract once and enforce it in validation.
- **Module boundary choice.** The five-module split isn't sacred; it's the one ModularRSI found worked for coding/terminal agents. Retrieval-heavy agents, for instance, may need Retrieval as a sixth module.
- **Integration edits ≠ module edits.** Cap the integration stage's power — otherwise it becomes a covert monolithic edit round and defeats attribution.
- **Cross-module deficiencies.** Some failures live between modules (e.g. bad Observation Mgmt output confuses Task Completion). The paper handles these by allowing paired-module edits when contrastive analysis names two modules together.

## Sources

- Paper: *ModularRSI: Modular and Generalizable Recursive Harness Self-Improvement* — Wu et al., 2026 — [arXiv:2609.14857](https://arxiv.org/abs/2609.14857).
