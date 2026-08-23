# Identity-Preserving Group Image Generation (WithEveryone)
*Depth — scaling identity-preserving image generation to many specified people via layout-grounded ID supervision and pre-image identity representation forcing.*

**TL;DR:** Identity-preserving image generation is unreliable once a scene must contain many specified people: models must *bind* each reference to a distinct person and location, and training-time ID losses rely on brittle embedding-based face matching over noisy predicted faces. **WithEveryone** handles up to **10 reference identities** by injecting each identity as an addressed token, predicting a structured **identity–layout plan**, and rendering the plan as a visual condition. Its main loss — **Layout-Grounded ID Loss** — uses annotated face regions to supervise the intended identity directly, avoiding embedding matching. On an identity-disjoint benchmark: face similarity **0.462 → 0.499** vs. GPT-Image-2, copy-paste artifact rate **0.169 → 0.055**, coverage of requested identities **97.3%** with only **2.8%** duplicates.

**Prereqs:** [../fundamentals/attention.md](../fundamentals/attention.md)
**Related:** none yet in the graph on diffusion identity conditioning.

---

## What it is

A conditioning + supervision recipe for multi-person diffusion generation. Two structural pieces:

- **Addressed identity tokens.** Each reference identity is injected as a distinguishable token in the conditioning stream, so the model can bind identity ↔ layout region unambiguously.
- **Structured identity–layout plan.** Before pixel synthesis, the model predicts *who goes where* (identity ↔ region), then that plan is rendered as an explicit visual condition (mask/layout image) alongside text.

## How it works

Two training-time ingredients:

- **Layout-Grounded ID Loss.** Instead of matching predicted faces to reference identities via embeddings (noisy when several similar faces are present), the loss uses annotated face regions of the plan: identity *k* is supervised only inside the region the plan assigned to identity *k*. Matching is deterministic and cheap.
- **ID Representation Forcing.** The model must produce a predictive representation for each identity *before* image synthesis. This forces the network to commit to a bound identity per slot rather than sampling identities into arbitrary faces later.

Inference: user supplies references + optional layout hints → model predicts identity–layout plan → conditions synthesis on plan + reference tokens.

## Why it matters

"Give me these ten people in one scene" has been the ID-preservation era's blind spot; naive multi-reference conditioning devolves into copy-pasting one face across the frame. The core insight — **plan the identity–layout binding first, condition on the plan** — echoes planner-executor patterns in agents: hard multi-object grounding wants an explicit intermediate representation, not end-to-end sampling. Copy-paste artifact rate dropping 3× is the clearest concrete win.

## Gotchas & tricks

- The plan is only as good as the layout predictor: bad plans (overlapping regions, missing identities) propagate straight into the output.
- Annotated face regions are needed at training time; scaling to novel identity types (children, occluded profiles, side-view) requires broadening the annotation distribution.
- Copy-paste rate reduction can come at the cost of *lower* per-face similarity if the layout planner leaves too little pixel budget per identity — the 0.462 → 0.499 improvement suggests they cleared that tradeoff, but budget matters at inference.

## Sources

- Paper: *WithEveryone: Unified Planning and Identity Grounding for Group Image Generation* — Xu, Wang, Cheng et al. (Fudan), 2026 — [arXiv:2608.20336](https://arxiv.org/abs/2608.20336)
