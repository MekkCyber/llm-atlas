# CaSKG — Counterfactual-Causal Skill Graph Retrieval
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** For agents with reusable skill libraries, prompting the full library is expensive, vector retrieval is workflow-blind (treats skills as independent), and existing graph retrieval only works if the edges are trustworthy. **CaSKG** builds a directed candidate graph from semantic/lexical/IO/structural evidence, then **calibrates each edge with counterfactual textual probes** — remove, substitute, or reorder the skill pair and observe the effect — aggregates with Bayesian smoothing, and publishes a state-filtered weighted graph for task-conditioned expansion. Top task score in all 12 model × benchmark combinations on ALFWorld ID-140 and ScienceWorld U211; six-model macro-average ScienceWorld 72.62 → 80.50 over Graph-of-Skills, ALFWorld 80.01% → 86.79% success, with fewer environment steps.

**Prereqs:** [README.md](README.md)
**Related:** [skill-evolution.md](skill-evolution.md)

---

## What it is

Agent skill libraries are a memory-retrieval problem. Three existing approaches:

- **Full-library prompting** — preserves coverage, pays a heavy context tax on every step.
- **Vector retrieval** — compact, but each skill is retrieved independently; loses the *procedural context* (which skills tend to run before/after each other).
- **Graph retrieval** (e.g. Graph-of-Skills) — encodes procedural edges, but the edges are usually inferred from noisy co-occurrence or LLM heuristics and don't reliably capture *causal* dependencies.

CaSKG is a **graph-retrieval** approach with a serious answer to the edge-trust problem: it treats edge weights as quantities to be *calibrated offline* before the graph is used online.

## How it works

**Offline graph construction:**

1. **High-recall candidate graph.** For each pair of skills, aggregate four evidence sources into a candidate edge score:
   - Semantic similarity of skill descriptions.
   - Lexical overlap of skill code / preconditions.
   - IO alignment (skill A's outputs match skill B's inputs).
   - Structural evidence from execution traces.
2. **Refinement.** Repair evidence from historical failures plus an optional LLM judge further refines candidate scores. This is still uncalibrated — high recall, medium precision.
3. **Counterfactual probes.** For each candidate directed edge (A → B), run three textual counterfactuals through an LLM:
   - **Remove** — does removing B after A change the task outcome?
   - **Substitute** — does replacing B with a lookalike change it?
   - **Reorder** — does swapping A and B change it?
   Aggregate the probes into evidence for/against the edge.
4. **Bayesian smoothing.** Combine probe evidence with priors and per-edge sample counts to produce a calibrated posterior edge weight.
5. **Publish.** State-filtered weighted graph, ready for online use.

**Online use:**

- Given the current state and task, seed retrieval from semantically-relevant skills, expand along high-weight causal edges (task-conditioned), and return a compact skill set. No change to the downstream agent policy.

## Why it matters

- **Wins across models and benchmarks.** Top score in 12/12 model × benchmark combinations — a rare completeness.
- **Reduces environment steps too.** Better retrieval → less flailing → shorter successful trajectories.
- **Compute is spent offline.** Once the graph is built, per-query retrieval is cheap. Suitable for skill libraries that grow slowly.
- **Counterfactual probing is a general primitive.** The offline-calibration idea (build noisy edges → probe → smooth → publish) is portable to other agent-memory graphs (concept graphs, tool graphs).

## Gotchas & tricks

- **Probe LLM cost is real, but offline.** Every candidate edge needs 3 probes × several runs; budget it into library-build time rather than per-query.
- **Direction-conditioned probes matter.** (A → B) and (B → A) are separately calibrated; the paper's edges are asymmetric.
- **State filtering is non-optional.** Even a well-calibrated graph over-retrieves without task-conditioned expansion — the state filter is what makes retrieval compact.
- **Doesn't grow with the skill library out of the box.** Adding new skills requires re-probing their neighborhood; incremental updates are needed for large libraries.

## Sources

- Paper: *CaSKG: Counterfactual-Causal Skill Graphs for Scalable Agent Skill Retrieval* — Li et al. (Jilin / Yale), 2026 — arXiv:2608.25500.
