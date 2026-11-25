# Automatic Tool Generation
*Depth — agents that write and reuse new tools at runtime instead of relying on a frozen library.*

**TL;DR:** A frozen handcrafted tool library is the main failure mode of multimodal search agents in the wild — new query types, evidence formats, and failure modes arrive constantly. ToolSmith (2026) proposes a loop in which the agent composes a new tool from existing primitives, evaluates it on the current query, and promotes successful tools into a reusable library. Larger backbones reap larger gains — the mechanism scales with the agent model's own composition ability.

**Prereqs:** [README](README.md)
**Related:** [frontier-as-policy](frontier-as-policy.md)

---

## What it is

A **library-growth loop** that sits on top of a small, fixed base library. When an existing tool doesn't cover a query's needs, the agent proposes a new tool — typically a short code module, prompt chain, or function composition built from the base primitives — tests it, and if it works, stores it for future reuse across tasks.

## How it works

Four stages, usually implemented as prompt-driven subroutines:

1. **Need detection** — the agent recognizes it cannot answer with current tools (missing modality coverage, insufficient filtering, etc.).
2. **Compose** — generate a new tool as code or sub-prompt chain assembled from the base library's primitives. The agent is prompted with the base tool schemas and asked to synthesize.
3. **Evaluate** — run the candidate on the current query and a small held-out probe; a VLM-as-a-Judge protocol scores correctness/relevance.
4. **Promote** — on success, write the tool into the library (with a short description) so later queries can retrieve and reuse it.

ToolSmith evaluates across 3 in-domain and 10 out-of-domain multimodal search benchmarks. Larger agent backbones generate better tools, and reuse fraction across tasks is first-class in evaluation.

## Why it matters

Handcrafted tool libraries are the brittlest layer of long-running agents — they decay the moment a new site/API/evidence format shows up. Making the agent accrete its own library is a plausible route to agents that generalize out-of-distribution without a human maintainer in the loop. The OOD gain in ToolSmith is the key evidence; in-domain gains alone would be unconvincing.

## Gotchas & tricks

- **Promotion gating is where this fails.** A single success on the current query is a weak signal; the library bloats with task-specific tools that don't reuse. Held-out probes or usage-count thresholds help.
- **Composition quality tracks backbone size** — smaller models compose brittle tools, so the loop pays off most on frontier agents.
- **Catalog retrieval matters at scale.** When the library grows into hundreds of tools, retrieving the right subset for the next query is itself a non-trivial sub-problem (embedding-based or hierarchical catalogs).
- The paper evaluates with an LLM judge, so absolute numbers should be read with that caveat.

## Sources

- Paper: ToolSmith: An Automatic Tool-Generation Framework for Multimodal Search Agents — Huang et al., 2026 — [arXiv:2611.31198](https://arxiv.org/abs/2611.31198)
- Project page: https://threesr.github.io/toolsmith/
