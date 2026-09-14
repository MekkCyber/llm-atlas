# Code World Models (recursive)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Represent a world as an **executable program** — objects, physics, and relationships as code — reconstructed by an LLM. Recursive Code World Models (RCWM) extend this by giving the constructor a **recursive solver**: sketch the whole scene, recurse on unresolved parts, then revisit the whole. The result is a compositional Recursive Scene Program (RSP) whose fine-scale structures each get their own perception-edit loop.

**Prereqs:** [README](README.md)
**Related:** [../multimodal/README](../multimodal/README.md)

---

## What it is

Code world models represent state as programs — objects as function calls, scene graphs as compositional code — so the LLM can *reason over*, *execute*, and *edit* the world instead of predicting raw pixels. The representation is editable and inspectable, which is what makes it attractive for agents.

The recursive-solver variant addresses the failure mode of flat code-world-models on complex scenes: single-shot construction under-resolves nested detail. RSPs are code that calls itself; the solver mirrors that structure.

## How it works

- **Recursive Scene Program (RSP).** A scene is a top-level function whose body constructs sub-assemblies, each itself an RSP call. This gives a natural tree over the world.
- **Global-local-global solver.**
  1. **Establish the whole** — from a single reference image, sketch the coarse scene as an RSP with placeholders for unresolved parts.
  2. **Recurse on unresolved parts** — each placeholder triggers another solver call: perceive the sub-region, edit its sub-RSP, refine.
  3. **Revisit the whole** — with children resolved, refine the parent to preserve global geometry and cross-part relationships.
- **Perception at every level.** The solver doesn't just prompt from the original image; each recursion re-perceives its own sub-region, so fine-scale details are seen at higher effective resolution.
- **Executable output.** The final RSP is code — you can run it to render the scene, edit any node to change the scene, or introspect the tree to reason about the world.

## Why it matters

Flat code-world-models lose structural fidelity at scale — one prompt can't attend equally to every corner of a complex scene. Recursion is the classical answer that had been missing. It maps cleanly to the divide-and-conquer patterns agent systems already use for long-horizon planning, so a *world* representation can now match a *plan* representation in expressiveness.

## Gotchas & tricks

- **Recursion termination is a real risk.** Without a resolution budget or a "fine enough" check, the solver can recurse indefinitely on textured regions. Cap depth; add a stopping heuristic based on residual reconstruction loss.
- **The final revisit-whole pass is essential.** Without it, children may be locally correct but globally inconsistent (misaligned or scaled). The paper's ablations show the "final global" step is what preserves scene-wide geometry.
- **Language of code matters.** Domain-specific scene DSLs (blocks, primitives, transforms) work better than raw Python — the LLM writes cleaner, more editable programs.
- **Perception ≠ generation.** The solver's perception step should be a *scoped* re-perception on the sub-region, not another pass over the whole image. Otherwise the recursion is wasted.
- **Editability is a use, not a bonus.** Design the RSP grammar so downstream agents can plug in edits (add/remove objects, transform sub-trees) without regenerating.

## Sources

- Paper: *Recursive Code World Models: Building Complex Worlds through Recursive Scene Programs* — Li, Liao, Zhu (Georgia Tech), 2026 — https://arxiv.org/abs/2609.11499
