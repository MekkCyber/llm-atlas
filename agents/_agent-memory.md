# Agent Memory

*Taxonomy — how an agent stores and retrieves state across actions, turns, sessions, and principals.*

**TL;DR:** "Agent memory" is a moving target because it conflates several distinct concerns: how *long* something is remembered (action / session / lifetime), *who* can read or write it (single-principal / shared), and *what* it is (facts / preferences / tool experience / world-state observations). The modern designs are converging on splitting these axes explicitly — separate stores per scope, separate governance per principal, separate retrieval policies per data type — rather than a single retrieval-augmented blob.

**Related taxonomies:** [_rag](_rag.md)
**Depth files covered here:** [hierarchical-agent-memory](hierarchical-agent-memory.md) · [memory-governance](memory-governance.md) · [governed-memory](governed-memory.md)

---

## The problem

A monolithic "memory" — append every observation + every tool result + every user message to one vector store, retrieve top-K per query — fails on every realistic deployment:

- Across sessions, working preferences leak into the long-term profile.
- Across principals (multi-user, multi-org), facts leak across authorization boundaries.
- After explicit deletion requests, retrieval still surfaces the content via paraphrase.
- For embodied agents, the agent "remembers" world-state it never actually observed.

These failure modes are not bugs in the retriever — they're a category error in the design (treating memory as one homogeneous store).

## The shared pattern

The modern variants all add *structure* to the memory store, along one or more of these axes:

- **Temporal scope.** Lifetime memory (persistent across sessions), working memory (active during the current task), action-step memory (intra-trajectory state).
- **Principal scope.** Per-user, per-organization, shared-but-controlled.
- **Content type.** User profile, world-state observations, tool/skill traces, reasoning scratchpads.
- **Governance metadata.** Quality, confidence, lifecycle stage, verifier outcomes, conflict signals.

## Variants

| Technique | Key idea | Main tradeoff | When it wins |
| --- | --- | --- | --- |
| Flat RAG-as-memory | One vector store, append + top-K retrieve | No structure, no governance | Single-user, small scope |
| [hierarchical-agent-memory](hierarchical-agent-memory.md) | Split into long-term / working / tool memory by temporal scope | Three stores to maintain | Multi-turn personalization |
| [governed-memory](governed-memory.md) | Tag each entry with confidence / lifecycle / conflicts; precision-oriented retrieval | Metadata maintenance | Long-running agents with quality drift |
| [memory-governance](memory-governance.md) (eval) | Benchmark utility / access-control / forgetting jointly | It's an eval, not a technique | Diagnosing shared-deployment memory |
| Observation-grounded memory (ObsMem) | Tag entries with visibility metadata; embodied agents won't assert un-observed state | Needs the trace pipeline | Embodied / partial-observability settings |

## How to choose

Default to **flat RAG-as-memory** while the deployment is single-user and small. The moment you have any of (multi-turn revisions, multiple users sharing one agent, long-running deployment with quality drift, embodied partial observability), pick the structured variant that addresses your specific failure mode rather than upgrading the retriever. Most production stacks now combine hierarchical *scoping* (when) with governance *metadata* (quality) on each store.

## Adjacent but distinct

- **[_rag](_rag.md)** — RAG is the retrieval substrate; memory is the *system* that decides what to store, when, who can read it, and when to delete. RAG is read-only over a static corpus; memory is read-write per principal.
- **Long-context** — keeping everything in the context window is an alternative for short horizons but doesn't address governance or multi-principal sharing.

## Sources

- *MemSlides: A Hierarchical Memory Driven Agent Framework for Personalized Slide Generation* — Jin et al., 2026 — https://arxiv.org/abs/2606.17162
- *GateMem: Benchmarking Memory Governance in Multi-Principal Shared-Memory Agents* — Ren et al., 2026 — https://arxiv.org/abs/2606.18829
- *GeneralVLA-2: Geometry-Aware Reconstruction and Governed Memory for Robot Planning* — Wang et al., 2026 — https://arxiv.org/abs/2606.17480
- *WorldLines: Benchmarking and Modeling Long-Horizon Stateful Embodied Agents* — Su et al., 2026 — https://arxiv.org/abs/2606.18847
