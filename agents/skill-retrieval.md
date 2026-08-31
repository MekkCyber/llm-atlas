# Skill retrieval via counterfactual-causal graphs
*Depth — CaSKG's counterfactual edge calibration for large agent skill libraries.*

**TL;DR:** As reusable skill libraries grow, similarity-based retrieval (cosine over embeddings) breaks — surface similarity does not equal downstream utility. CaSKG builds an offline skill graph from semantic + lexical + structural evidence, then **counterfactually probes** every edge — "would the downstream task succeed if this edge were followed?" — and reweights the graph accordingly. Retrieval walks a graph whose edges reflect causal utility.

**Prereqs:** [README.md](README.md)
**Related:** [skill-evolution.md](skill-evolution.md)

---

## What it is

A retrieval system for LLM agents that must select relevant skills from a library of thousands of entries. CaSKG replaces heuristic similarity ranking with a graph whose edges are **causally calibrated** by test-time counterfactual probes: for each candidate edge, the system simulates whether picking that skill actually helps solve the downstream task; the empirical success rate becomes the edge weight.

## How it works

Two phases:

1. **Offline graph construction.**
   - Nodes: skills in the library.
   - Edges: derived from three evidence sources — semantic similarity (embedding distance), lexical overlap (shared identifiers / parameters), and structural relations (call graph, argument type compatibility).
   - Result: a dense candidate graph, uncalibrated.

2. **Counterfactual probing.**
   - For each edge (or a sampled subset), run a probe task where one arm uses the edge and the other does not.
   - Score the outcome (task success / cost).
   - Reweight the edge with the causal effect estimate — high positive effect → strong edge; near-zero or negative → prune.

At runtime, the agent retrieves by walking the calibrated graph rather than reranking by raw embedding similarity.

## Why it matters

Skill libraries scale to thousands of entries; naive retrieval regresses to a few over-represented skills or hallucinates matches. Counterfactual calibration is a general trick — anywhere a heuristic graph is being used for retrieval, edge-level causal probes can turn it into a calibrated one.

Reported results across six backbones on standard agent benchmarks:
- **ScienceWorld:** 72.62 → 80.50
- **ALFWorld:** 80.01% → 86.79%

## Gotchas & tricks

- Probing is expensive — do it once offline, per graph revision, not per query.
- Probe task selection matters: sample tasks that stress edges, not tasks where any skill would succeed.
- Combine with a curated wiki (see [skill-evolution.md](skill-evolution.md)) to keep the skill library itself high-quality; garbage-in-graph, garbage-out.
- The causal effect estimator is only as good as the probe's outcome measurement — vague success signals give noisy edge weights.

## Sources

- Paper: *CaSKG: Counterfactual-Causal Skill Graphs for Scalable Agent Skill Retrieval* — Li, Gao, Ding, Chen, Wu, Chang, 2026 — [arXiv:2608.25500](https://arxiv.org/abs/2608.25500)
