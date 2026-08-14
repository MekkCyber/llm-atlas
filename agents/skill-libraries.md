# Skill Libraries
*Depth — packaging agent procedural knowledge as loadable skills, and how those libraries scale under a context budget.*

**TL;DR:** A "skill" is a reusable procedural artifact — some code, some documentation, and the contract that links them (inputs, outputs, invoked verifiers, prerequisite skills). Agents load skills at inference time and execute them mid-turn. As skill libraries grow past a few hundred entries, naive loading exceeds the context budget; the field has converged on treating the library as a **graph** and compressing at the section level while preserving procedural contracts. Introduced in scaled form by SkillZip (2026).

**Prereqs:** [../agents/README.md](README.md)
**Related:** [../inference/README.md](../inference/README.md)

---

## What it is

A skill library is the model's *procedural memory*. Each skill bundles:

- **Code** — the deterministic operation the agent will execute (a shell recipe, a template, a plotting function).
- **Documentation** — natural-language guidance on when and how to invoke the skill.
- **Contract** — declared inputs/outputs, verifiers that check the skill ran correctly, and prerequisite skills that must be loaded first.

Anthropic Skills, Cursor rules, Codex skills, and MCP-server tool packs are all specific instantiations of the same idea. The problem is scaling — production libraries reach 10⁴–10⁵ entries, and inference-time context is a shared, scarce resource.

## How it works

The core operations:

1. **Retrieval.** Rank skills by relevance to the current task (embedding search over documentation is the common default; SkillZip augments with contract-graph reachability).
2. **Contract expansion.** For each retrieved skill, walk the prerequisite graph so every dependency is loaded together — otherwise the skill fails at execution time.
3. **Context-budget compression.** Fit the selected subgraph into the model's context by dropping non-essential sections while preserving every contract edge (declared IO, verifier references, prereq links). Naïve token-count trimming breaks skills silently; contract-preserving compression breaks them loudly (or not at all).
4. **Execution + verification.** The agent invokes the skill; the declared verifier confirms it ran. Failed verifiers surface into the trajectory instead of failing silently.

SkillZip formalizes step 3 as **section-level graph compression** and reports a 3.46× compression ratio at 100k-skill scale with high verifier-reachability preservation.

## Why it matters

- Skill libraries are how agents get customized without fine-tuning — the practical differentiator between deployments of the same underlying model.
- Context budget is the ceiling on how many skills a run can compose. Every unit of headroom won by contract-preserving compression is a unit spent on task tokens instead.
- Structured contracts make skill libraries *auditable*: you can reason about what an agent might do next from the loaded contract subgraph.
- Adjacent to the runtime-contract framing ([../safety/runtime-contract.md](../safety/runtime-contract.md)) — contract-verified skill execution is one leg of an evidence chain.

## Gotchas & tricks

- **Contract-preserving is not lossless.** Compression can drop rare-but-critical prose sections while keeping the contract edges intact. Verifier coverage catches this; taste doesn't.
- **Retrieval ranks by "sounds relevant," not by "will actually run."** Add prerequisite closure to the ranking, not just document similarity.
- **Skills poison memory.** A skill can be [prompt-injected](../safety/indirect-prompt-injection.md) via its documentation strings; sign or hash the library and gate updates.
- **The right unit is the section, not the skill file.** Skills are heterogeneous — the same file can carry both invocation examples and edge-case caveats — and section-level compression captures that granularity.

## Sources

- Paper: *SkillZip: Contract-Preserving Graph Compression for Scalable Agent Skill Libraries* — Tan, Wang, Liu, Xu, Yuan, Zhu, Zhang, 2026 — [arXiv:2608.05604](https://arxiv.org/abs/2608.05604) — canonical scalable-skill-library formulation and the section-level graph-compression method.
