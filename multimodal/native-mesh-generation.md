# Native Mesh Generation with Flow Matching
*Depth — generating triangle meshes directly, rather than extracting them from an implicit field.*

**TL;DR:** Most 3D generators produce an implicit surface (SDF, NeRF, occupancy grid) and mesh it afterwards, which is slow and produces topologically noisy geometry. Native mesh generation trains a model whose output space *is* a mesh — vertices and edges jointly — under a **flow-matching** objective. **Meshy T2** demonstrates this at interactive speed with controllable face budgets, and gets multi-part decomposition for free because parts fall out as connected components of the generated graph.

**Prereqs:** [text-conditioning-scaling.md](./text-conditioning-scaling.md)
**Related:** [pt-flow.md](./pt-flow.md)

---

## What it is

A category of 3D generative models where the network directly outputs mesh geometry (vertices + connectivity) instead of a scalar/vector field that must be meshed. The training objective of choice is flow matching, which is well-suited to generating discrete geometric structures because the target distribution can be a mixed continuous-discrete object (vertex positions + adjacency).

## How it works

**Output representation.** A mesh is `(V, E)` — vertex positions plus an edge/face set. Native generators emit both:

- **Vertex positions** as continuous 3-vectors.
- **Connectivity** as a set of edges or triangles that can be encoded as either adjacency logits or a graph-generation head.

**Flow-matching objective.** Condition a velocity field on a text prompt (or other input); train it so that integrating from noise produces a valid mesh. Because vertices and connectivity are learned jointly, the model learns coherent topology, not just point positions.

**Controls.**
- **Face budget** as a scalar input — the model targets a mesh of that size.
- **Text prompt** for content control.

**Multi-part decomposition (emergent).** With connectivity generated jointly with vertices, distinct parts of an asset show up as separate connected components — no downstream part segmenter needed.

## Why it matters

- Compact, topologically clean meshes at interactive speed — usable directly in DCC tools without an implicit-surface artifact cleanup pass.
- Face budget as a first-class control gives an operator a knob that implicit methods can only approximate via remeshing.
- Concrete evidence that flow matching applies well beyond continuous domains — meshes are a mixed continuous/discrete output and the objective still trains cleanly.

## Gotchas & tricks

- The connectivity head is the sensitive part; small mistakes produce non-manifold meshes. Manifold-preserving losses or post-generation topology checks are usually needed.
- Flow matching on the mixed continuous-discrete target requires careful noise schedules — a single continuous schedule for the vertices is fine, but connectivity typically needs a discrete corruption process.
- Cross-part consistency (e.g., a hinge that must span two parts) is harder for native mesh generation than for implicit methods.

## Sources

- Paper: *Meshy T2: Fast Native Mesh Generation with Flow Matching* — Xu et al. (Meshy AI), 2026 — [arXiv:2607.28675](https://arxiv.org/abs/2607.28675)
