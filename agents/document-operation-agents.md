# Document-Operation Agents (DocOps)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** DocOps is a deterministically verifiable benchmark for autonomous agents on complex document workflows (edit, restructure, cross-doc consistency). Decomposes real-world ops into atomic dimensions and escalating workflow complexity. Frontier agents still collapse on highly coupled long-range tasks, with three consistent failure modes: **long-term state-tracking collapse**, **shallow semantic verification**, and **destructive editing of structural metadata**. The paper's contribution is (a) the eval and (b) the failure-mode taxonomy — both directly actionable for agent developers.

**Prereqs:** [README.md](./README.md)
**Related:** [../evaluation/README.md](../evaluation/README.md), [../safety/cot-monitoring.md](../safety/cot-monitoring.md)

---

## What it is

Document manipulation is the workhorse of real workspace agents (M365 Copilot, Notion AI, coding-agent doc edits). But most "agent benchmarks" grade with an LLM-judge, which conflates the agent's failure modes with the judge's biases and can be gamed by output style. DocOps replaces the judge with **deterministic verifiers over document state**: the answer key is a structural-metadata check on the resulting document, not a natural-language rubric.

Combined with a **hierarchical taxonomy** that starts at atomic ops (rename a heading, delete a paragraph) and escalates to composite workflows (restructure a section while preserving cross-references), the benchmark can attribute agent failures to specific complexity axes rather than "the agent got it wrong."

## How it works

**Taxonomy.** Atomic operations along multiple dimensions (structural, semantic, formatting, cross-reference). Composite workflows chain atomic ops under complexity levels (short-range vs long-range coupling).

**Verifier.** Deterministic check on the resulting document's state — does the final DOM/structure match the expected result? Passes iff the check passes; no LLM judge in the loop.

**Evaluated stacks.** Representative closed- and open-source models under various agentic harnesses. Even the strongest configurations collapse on highly coupled long-range tasks.

**Failure taxonomy.** Three named modes surfaced by the eval:
- **Long-term state-tracking collapse.** The agent forgets constraints established earlier in the workflow.
- **Shallow semantic verification.** The agent's self-checks look at surface form, not meaning — an edit that looks right but changes the underlying content passes.
- **Destructive editing of structural metadata.** The agent modifies underlying XML/DOM/ref structure in ways that break cross-references or formatting, even when the visible text is correct.

## Why it matters

- **Deterministic verifier removes judge bias.** DocOps is one of the first agent evals where the score is a mechanical check, not an LLM judgment. Directly comparable across runs and models.
- **Failure-mode taxonomy is actionable.** Agent developers can target one of the three named failure modes; the eval will tell them whether the fix worked without the judge-drift problem.
- **Bar for workspace-agent claims.** Any product claiming "our agent handles complex document workflows" now has a concrete benchmark to report against.

## Gotchas & tricks

- Deterministic verifiers are strict — an edit that a human would accept as "close enough" may fail the check. This is by design for evaluation but doesn't reflect user tolerance.
- The three failure modes are specific to *document* workflows; broader agent-safety failure taxonomies still need work.
- Ground-truth document states are hand-curated; the benchmark's scale is bounded by that curation cost.

## Sources

- Paper: *DocOps: A Verifiable Benchmark for Autonomous Agents in Complex Document Operations* — Jiang, Cao, Yan et al., CASIA / Baidu, 2026 — [arXiv:2607.19865](https://arxiv.org/abs/2607.19865)
