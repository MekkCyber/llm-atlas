# Code World Models
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Represent a world (typically a 3D scene) as an **executable program** rather than a learned latent, and use a VLM coding agent to synthesize, render, and iteratively refine the program against a reference image. Recursive Code World Models (RCWM) introduce a **recursive scene-program** representation solved with a global-local-global schedule, so wide-scene geometry stays consistent while fine detail is added by inner passes. Outperforms prior code-based image-to-scene methods for reconstruction from a single reference image.

**Prereqs:** [../multimodal/README](../multimodal/README.md)
**Related:** [README](README.md)

---

## What it is

Two design choices define a "code world model":
1. The world is stored as *code* — a program in a rendering DSL, a scene graph, or Python calls to a renderer. Executing the program materializes the world; editing the program edits the world.
2. The **agent** that constructs and refines the code is a VLM: it takes a reference image, produces the program, renders it, compares the render to the reference, and edits.

This contrasts with **neural world models** (latent-space video/scene predictors) which encode the world in weights + activations. Code representations gain hard structural guarantees (a chair inserted with a `place_chair(...)` call *is* a chair) and interpretability, at the cost of an outer search loop.

## How it works

**Recursive scene programs (RCWM).**
- The scene is a tree of nested program units — outer units express global geometry (rooms, roads), inner units express local structure (furniture, trees), and inner-inner units express fine detail (books on a shelf).
- Each unit is a callable in the rendering DSL; its call arguments are 3D position, orientation, and style.

**Global-local-global recursion.**
1. *Global pass 1.* Coding agent sketches the top-level unit — coarse layout that matches the reference image at low frequency.
2. *Local pass.* For each child unit, recurse: propose child code, render, compare, edit. This pass fills structure at intermediate scales without touching global geometry.
3. *Global pass 2.* After locals return, run a top-down consistency pass that adjusts global params to reconcile any local-scale corrections that broke wide-scene geometry.

**Loop.** Compare render vs reference with a perceptual similarity metric (LPIPS, CLIP-similarity, VLM judge). If the delta exceeds threshold, the agent proposes a targeted edit at the appropriate scale.

## Why it matters

- **Structure by construction.** A `place_chair(...)` call cannot produce a half-chair; a neural latent can and does.
- **Editability.** Changing "chair" to "sofa" is a code edit, not a re-encoding of the world.
- **VLM agent alignment.** Coding-agent techniques (tool use, self-critique, program repair) transfer directly to world building.
- **Complementary to neural world models.** Neural models excel at plausibility and texture; code models excel at hard constraints. Hybrid stacks are plausible.

## Gotchas & tricks

- **Search cost.** Iterative refine loops are slow; cache renders and reuse local subtree renders across iterations.
- **Reference metric matters.** Pixel-level losses over-penalize style mismatch; use a VLM judge or CLIP-similarity for the fitness signal.
- **DSL expressiveness.** Too narrow a DSL forces the agent to approximate outside its language; too broad and search blows up. Start with a curated primitive set and expose escape hatches for procedural detail.
- **Global-local mixing.** Skipping global pass 2 leaves scenes locally-plausible but globally-inconsistent (rooms that don't tile). Always close the recursion.
- **Not a general "world model" replacement.** Doesn't predict physics or long-horizon dynamics — this is a *reconstruction* world model, not a rollout model.

## Sources

- Paper: *Recursive Code World Models: Building Complex Worlds through Recursive Scene Programs* — Li, Liao, Zhu, 2026 — [arXiv:2609.11499](https://arxiv.org/abs/2609.11499)
- Related: neural / video-based world models (see `../multimodal/video-world-models.md` for the training-free-control counterpart).
